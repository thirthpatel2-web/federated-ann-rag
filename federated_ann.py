"""Step 2: what does it cost to search a corpus that is split into silos?

Setup
  * One shared embedding model for every silo (the simple, optimistic case).
  * Silos: "topic" (spherical k-means on the embeddings) or "random" (a control),
    S = 2, 5, 10, each built with 3 seeds.
  * One index per silo: Flat (exact), HNSW (efSearch 16/64/160), IVF-Flat (nprobe 1/8)
    and IVF-PQ with exact re-ranking of candidates (IndexRefineFlat), so that scores from
    different silos are comparable when merged.
  * Federated search queries every silo and merges candidates by score into a top-10.
    How many results each silo may contribute:
      full     every silo returns 10 (no budget)
      equal    every silo returns ceil(10 / S)
      routed   a total of 10 results shared across silos in proportion to how similar the
               query is to each silo's centroid (softmax, temperature 0.05)

Metrics (k = 10), averaged over 1,000 questions and over seeds
  both_gold   share of questions with BOTH gold paragraphs in the top-10, with a 95%
              bootstrap confidence interval over questions (first seed)
  nn_recall   overlap with the exact top-10 of one central index
  work        distance computations per query, summed over all silos (a hardware-free
              measure of search effort)
  colocated   share of questions whose two gold paragraphs are in the same silo
"""
import json
import math
from pathlib import Path

import faiss
import numpy as np

DATA, OUT = Path("data"), Path("results")
OUT.mkdir(exist_ok=True)
K, SEEDS, TEMP = 10, [0, 1, 2], 0.05
faiss.omp_set_num_threads(4)

P = np.load(DATA / "passages.npy")
Q = np.load(DATA / "queries.npy")
questions = [json.loads(line) for line in open(DATA / "questions.jsonl", encoding="utf-8")]
gold = np.array([q["gold_pids"] for q in questions])  # every question has exactly 2 gold paragraphs
n, d = P.shape
nq = len(Q)


def make_silos(num_silos, how, seed):
    if num_silos == 1:
        return [np.arange(n)]
    if how == "random":
        return np.array_split(np.random.default_rng(seed).permutation(n), num_silos)
    km = faiss.Kmeans(d, num_silos, niter=20, seed=seed, spherical=True)
    km.train(P)
    _, assign = km.index.search(P, 1)
    return [np.where(assign[:, 0] == s)[0] for s in range(num_silos)]


def build(vectors, kind):
    if kind == "flat":
        return faiss.IndexFlatIP(d)
    if kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        return index
    nlist = max(4, int(math.sqrt(len(vectors))))
    if kind == "ivfflat":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:  # ivfpq + exact re-ranking of 4x candidates
        base = faiss.IndexIVFPQ(faiss.IndexFlatIP(d), d, nlist, 16, 8, faiss.METRIC_INNER_PRODUCT)
        index = faiss.IndexRefineFlat(base)
        index.k_factor = 4
    index.train(vectors)
    return index


def set_param(index, kind, param):
    if kind == "hnsw":
        index.hnsw.efSearch = param
    elif kind == "ivfflat":
        index.nprobe = param
    elif kind == "ivfpq":
        faiss.downcast_index(index.base_index).nprobe = param


def search_counting_work(index, kind, k):
    """Search all queries; return scores, local ids and distance computations per query."""
    if kind == "flat":
        scores, ids = index.search(Q, k)
        return scores, ids, index.ntotal
    stats = faiss.cvar.hnsw_stats if kind == "hnsw" else faiss.cvar.indexIVF_stats
    stats.reset()
    scores, ids = index.search(Q, k)
    work = stats.ndis / nq
    if kind == "ivfpq":
        work += 4 * k  # exact re-ranking of the k_factor * k candidates
    return scores, ids, work


def allocate(silos):
    """Routed budget: share K results across silos by query-to-centroid similarity."""
    cents = np.stack([P[ids].mean(0) for ids in silos])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)
    w = np.exp((Q @ cents.T) / TEMP)
    w /= w.sum(1, keepdims=True)
    raw = w * K
    k = np.floor(raw).astype(int)
    rem = K - k.sum(1)
    order = np.argsort(-(raw - k), axis=1)
    for i in range(nq):
        k[i, order[i, :rem[i]]] += 1
    return k  # shape (nq, S)


def merge(per_silo, budgets):
    """per_silo: list of (scores, global ids) each (nq, K); budgets: (nq, S) results allowed."""
    S = len(per_silo)
    scores = np.full((nq, S * K), -np.inf, dtype="float32")
    ids = np.full((nq, S * K), -1)
    for s, (sc, gi) in enumerate(per_silo):
        mask = np.arange(K)[None, :] < budgets[:, s:s + 1]
        scores[:, s * K:(s + 1) * K] = np.where(mask, sc, -np.inf)
        ids[:, s * K:(s + 1) * K] = gi
    order = np.argsort(-scores, axis=1)[:, :K]
    return np.take_along_axis(ids, order, axis=1)


def both_gold(top):
    hit = (top[:, :, None] == gold[:, None, :]).any(1)  # (nq, 2)
    return hit.all(1).astype(float)


def bootstrap_ci(x, reps=1000, seed=0):
    rng = np.random.default_rng(seed)
    means = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(reps)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


exact = faiss.IndexFlatIP(d)
exact.add(P)
_, exact_top = exact.search(Q, K)

CONFIGS = {"flat": [None], "hnsw": [16, 64, 160], "ivfflat": [1, 8], "ivfpq": [8]}
rows = []
for how, S in [("central", 1)] + [(h, s) for h in ("topic", "random") for s in (2, 5, 10)]:
    acc = {}  # (kind, param, mode) -> list over seeds of (both_gold vector, nn, work)
    coloc = []
    for seed in (SEEDS if S > 1 else [0]):
        silos = make_silos(S, how, seed)
        silo_of = np.empty(n, dtype=int)
        for s, ids in enumerate(silos):
            silo_of[ids] = s
        coloc.append(float(np.mean(silo_of[gold[:, 0]] == silo_of[gold[:, 1]])))
        routed = allocate(silos) if S > 1 else None
        for kind, params in CONFIGS.items():
            indexes = []
            for ids in silos:
                index = build(P[ids], kind)
                index.add(P[ids])
                indexes.append(index)
            for param in params:
                per_silo, work = [], 0.0
                for ids, index in zip(silos, indexes):
                    set_param(index, kind, param)
                    sc, local, w = search_counting_work(index, kind, min(K, len(ids)))
                    gi = np.where(local >= 0, ids[np.clip(local, 0, None)], -1)
                    per_silo.append((sc, gi))
                    work += w
                modes = {"full": np.full((nq, S), K)}
                if S > 1:
                    modes["equal"] = np.full((nq, S), math.ceil(K / S))
                    modes["routed"] = routed
                for mode, budgets in modes.items():
                    top = merge(per_silo, budgets)
                    nn = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(top, exact_top)]))
                    acc.setdefault((kind, param, mode), []).append((both_gold(top), nn, work))
    for (kind, param, mode), runs in acc.items():
        bg = [r[0].mean() for r in runs]
        lo, hi = bootstrap_ci(runs[0][0])
        row = {"silos": how, "S": S, "index": kind, "param": param, "budget": mode,
               "both_gold": round(float(np.mean(bg)), 4), "both_gold_seed_sd": round(float(np.std(bg)), 4),
               "ci95_seed0": [round(lo, 4), round(hi, 4)],
               "nn_recall": round(float(np.mean([r[1] for r in runs])), 4),
               "work_per_q": round(float(np.mean([r[2] for r in runs]))),
               "colocated": round(float(np.mean(coloc)), 4)}
        rows.append(row)
        print(json.dumps(row), flush=True)

with open(OUT / "results.json", "w") as f:
    json.dump(rows, f, indent=1)
print("saved", OUT / "results.json")
