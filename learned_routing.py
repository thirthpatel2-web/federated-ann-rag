"""Learned per-silo budget predictor (a small kPred-style network) vs centroid routing.

Same setting as routing_cost.py: 10 topic silos, exact search in each silo, a total budget of
K = 10 results shared across silos, silos with a budget of zero are not searched.

A small PyTorch MLP reads the query embedding plus its similarity to each silo centroid and
predicts how the K results should be shared across silos. It is trained with two kinds of labels:
  nn    share of the exact central top-10 that lies in each silo (nearest-neighbour labels)
  gold  share of the two gold paragraphs that lies in each silo (relevant-passage labels)

Training and model selection use HotpotQA questions that are NOT in the 1,000 test questions
(4,000 train, 1,000 dev). Results -> results/learned_routing.json
"""
import json
import random
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

import federated_ann as fa

DATA = Path("data")
K, S = fa.K, 10
torch.set_num_threads(4)


def extra_questions():
    """Embeddings and gold ids of 5,000 questions disjoint from the test sample (cached)."""
    cache = DATA / "extra_questions.npz"
    if cache.exists():
        z = np.load(cache)
        return z["Q"], z["gold"]
    rows = pq.read_table(DATA / "hotpot_distractor_validation.parquet").to_pylist()
    title_to_pid = {}
    for r in rows:
        for title in r["context"]["title"]:
            title_to_pid.setdefault(title, len(title_to_pid))
    random.seed(13)
    test_ids = {r["id"] for r in random.sample(rows, 1000)}  # same draw as prepare.py
    rest = [r for r in rows if r["id"] not in test_ids]
    random.Random(0).shuffle(rest)
    rest = [r for r in rest if len({title_to_pid[t] for t in r["supporting_facts"]["title"]}) == 2][:5000]
    gold = np.array([sorted({title_to_pid[t] for t in r["supporting_facts"]["title"]}) for r in rest])
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    Q = model.encode([r["question"] for r in rest], batch_size=128, normalize_embeddings=True,
                     convert_to_numpy=True).astype("float32")
    np.savez(cache, Q=Q, gold=gold)
    return Q, gold


def allocate(prob):
    """Share K results across silos in proportion to prob, largest-remainder rounding."""
    raw = prob * K
    k = np.floor(raw).astype(int)
    rem = K - k.sum(1)
    order = np.argsort(-(raw - k), axis=1)
    for i in range(len(prob)):
        k[i, order[i, :rem[i]]] += 1
    return k


def features(Qm, cents):
    return np.hstack([Qm, Qm @ cents.T]).astype("float32")


def train(X, Y, Xd, Yd, seed):
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                              torch.nn.Linear(256, S))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    X, Y, Xd, Yd = map(torch.from_numpy, (X, Y, Xd, Yd))
    best, best_state = 1e9, None
    for epoch in range(40):
        net.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 128):
            idx = perm[i:i + 128]
            loss = -(Y[idx] * torch.log_softmax(net(X[idx]), 1)).sum(1).mean()  # soft cross-entropy
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            dev = -(Yd * torch.log_softmax(net(Xd), 1)).sum(1).mean().item()
        if dev < best:
            best, best_state = dev, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def evaluate(budgets, per_silo, sizes, central_top):
    top = fa.merge(per_silo, budgets)
    searched = budgets > 0
    return {"both_gold": float(fa.both_gold(top).mean()),
            "nn_recall": float(np.mean([len(set(a) & set(b)) / K for a, b in zip(top, central_top)])),
            "silos_searched": float(searched.sum(1).mean()),
            "distance_computations": float((searched * sizes).sum(1).mean())}


def main():
    Qx, gold_x = extra_questions()
    Qtr, gtr, Qdv, gdv = Qx[:4000], gold_x[:4000], Qx[4000:], gold_x[4000:]
    central = faiss.IndexFlatIP(fa.d)
    central.add(fa.P)
    _, central_test = central.search(fa.Q, K)
    _, central_tr = central.search(Qtr, K)
    _, central_dv = central.search(Qdv, K)
    out = {}
    for seed in fa.SEEDS:
        silos = fa.make_silos(S, "topic", seed)
        silo_of = np.empty(fa.n, dtype=int)
        for s, ids in enumerate(silos):
            silo_of[ids] = s
        sizes = np.array([len(x) for x in silos])
        cents = np.stack([fa.P[ids].mean(0) for ids in silos])
        cents /= np.linalg.norm(cents, axis=1, keepdims=True)
        per_silo = []
        for ids in silos:
            index = faiss.IndexFlatIP(fa.d)
            index.add(fa.P[ids])
            sc, li = index.search(fa.Q, K)
            per_silo.append((sc, ids[li]))

        def labels(kind, top, gold):
            src = top if kind == "nn" else gold
            y = np.zeros((len(src), S), dtype="float32")
            for i, row in enumerate(src):
                for pid in row:
                    y[i, silo_of[pid]] += 1
            return y / y.sum(1, keepdims=True)

        res = {"centroid": evaluate(fa.allocate(silos), per_silo, sizes, central_test)}
        for kind in ("nn", "gold"):
            net = train(features(Qtr, cents), labels(kind, central_tr, gtr),
                        features(Qdv, cents), labels(kind, central_dv, gdv), seed)
            with torch.no_grad():
                prob = torch.softmax(net(torch.from_numpy(features(fa.Q, cents))), 1).numpy()
            res[f"learned_{kind}"] = evaluate(allocate(prob), per_silo, sizes, central_test)
        out[f"seed{seed}"] = res
        print(seed, json.dumps(res), flush=True)
    keys = ["both_gold", "nn_recall", "silos_searched", "distance_computations"]
    out["mean"] = {m: {k: float(np.mean([out[f"seed{s}"][m][k] for s in fa.SEEDS])) for k in keys}
                   for m in ("centroid", "learned_nn", "learned_gold")}
    json.dump(out, open("results/learned_routing.json", "w"), indent=1)
    print("mean", json.dumps(out["mean"], indent=1))


if __name__ == "__main__":
    main()
