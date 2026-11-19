"""Step 2: how much retrieval quality is lost when the corpus is split into silos
and each silo is searched with an approximate nearest neighbour (ANN) index?

Setup
  * One shared embedding model for every silo (the simple, optimistic case).
  * Silos are built two ways: "topic" (k-means on the embeddings, so each silo is
    a topic cluster) and "random" (a control).
  * Every silo gets its own index: Flat (exact), HNSW, or IVF-PQ.
  * Federated search queries every silo and merges the results by raw score.
    Each silo returns either the full k results ("full") or only ceil(k / S)
    ("budget"), which is what a cost-limited federation would do.

Metrics (k = 10)
  nn_recall   overlap with the exact top-10 of ONE central index (how close we are
              to perfect search)
  gold_recall share of the question's two gold passages found in the top-10
  both_gold   share of questions with BOTH gold passages in the top-10
              (HotpotQA questions need both to be answered)
  ms_per_q    search time per question, all silos together

Expectation worth checking first: with exact search in every silo, full per-silo k
and merging by raw score, federation should lose NOTHING (the merged top-10 is
the central top-10). Any loss must then come from ANN or from the budget.
"""
import json
import math
import time
from pathlib import Path

import faiss
import numpy as np

DATA, OUT = Path("data"), Path("results")
OUT.mkdir(exist_ok=True)
K = 10
faiss.omp_set_num_threads(4)

P = np.load(DATA / "passages.npy")
Q = np.load(DATA / "queries.npy")
questions = [json.loads(line) for line in open(DATA / "questions.jsonl", encoding="utf-8")]
gold = [set(q["gold_pids"]) for q in questions]
n, d = P.shape


def make_silos(num_silos, how, seed=0):
    """Return a list of arrays of passage ids, one array per silo."""
    if num_silos == 1:
        return [np.arange(n)]
    if how == "random":
        perm = np.random.default_rng(seed).permutation(n)
        return np.array_split(perm, num_silos)
    km = faiss.Kmeans(d, num_silos, niter=20, seed=seed, spherical=True)
    km.train(P)
    _, assign = km.index.search(P, 1)
    return [np.where(assign[:, 0] == s)[0] for s in range(num_silos)]


def build_index(vectors, kind, param):
    if kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
    elif kind == "ivfpq":
        nlist = max(4, int(math.sqrt(len(vectors))))
        index = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 16, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    index.add(vectors)
    if kind == "hnsw":
        index.hnsw.efSearch = param
    if kind == "ivfpq":
        index.nprobe = min(param, index.nlist)
    return index


def evaluate(top, exact_top):
    nn = np.mean([len(set(a) & set(b)) / K for a, b in zip(top, exact_top)])
    g = [len(set(t) & gs) / len(gs) for t, gs in zip(top, gold)]
    both = [gs <= set(t) for t, gs in zip(top, gold)]
    return nn, float(np.mean(g)), float(np.mean(both))


def timed_search(silos, kind, param, per_silo_k):
    """Federated search: build one index per silo, query every silo, then merge all
    candidates by raw score into one top-K. Only the searching is timed."""
    built = [(ids, build_index(P[ids], kind, param)) for ids in silos]
    t0 = time.perf_counter()
    all_scores, all_ids = [], []
    for ids, index in built:
        k_s = min(per_silo_k, len(ids))
        scores, local = index.search(Q, k_s)
        all_scores.append(scores)
        all_ids.append(np.where(local >= 0, ids[np.clip(local, 0, None)], -1))
    elapsed = time.perf_counter() - t0
    scores, ids = np.hstack(all_scores), np.hstack(all_ids)
    order = np.argsort(-scores, axis=1)[:, :K]
    return np.take_along_axis(ids, order, axis=1), elapsed


# Ground truth: exact search over one central index
exact_top, _ = timed_search([np.arange(n)], "flat", None, K)

CONFIGS = [("flat", None), ("hnsw", 16), ("hnsw", 64), ("ivfpq", 1), ("ivfpq", 8)]
rows = []
for how in ["topic", "random"]:
    for S in [1, 2, 5, 10]:
        if S == 1 and how == "random":
            continue
        silos = make_silos(S, how)
        for kind, param in CONFIGS:
            for mode in (["full"] if S == 1 else ["full", "budget"]):
                per_k = K if mode == "full" else math.ceil(K / S)
                top, secs = timed_search(silos, kind, param, per_k)
                nn, g, both = evaluate(top, exact_top)
                row = {"silos": how if S > 1 else "central", "S": S, "index": kind,
                       "param": param, "per_silo_k": mode, "nn_recall": round(nn, 4),
                       "gold_recall": round(g, 4), "both_gold": round(both, 4),
                       "ms_per_q": round(1000 * secs / len(Q), 3)}
                rows.append(row)
                print(row, flush=True)

with open(OUT / "results.json", "w") as f:
    json.dump(rows, f, indent=1)
print("saved", OUT / "results.json")
