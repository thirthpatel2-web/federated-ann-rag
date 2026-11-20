"""How much search does routing save if silos with a zero budget are not searched at all?

Exact search inside each silo, 10 topic silos, routed budget of 10 results (same as federated_ann.py).
Skipping a zero-budget silo cannot change the merged top-10, so quality equals the routed result;
this script measures the silos searched and distance computations per query, and the overlap with
the exact central top-10. Results -> results/routing_cost.json
"""
import json

import faiss
import numpy as np

import federated_ann as fa

out = {}
central = faiss.IndexFlatIP(fa.d)
central.add(fa.P)
_, central_top = central.search(fa.Q, fa.K)
for seed in fa.SEEDS:
    silos = fa.make_silos(10, "topic", seed)
    budgets = fa.allocate(silos)
    sizes = np.array([len(s) for s in silos])
    searched = budgets > 0
    per_silo = []
    for ids in silos:
        index = faiss.IndexFlatIP(fa.d)
        index.add(fa.P[ids])
        sc, li = index.search(fa.Q, fa.K)
        per_silo.append((sc, ids[li]))
    top = fa.merge(per_silo, budgets)
    nn = np.mean([len(set(a) & set(b)) / fa.K for a, b in zip(top, central_top)])
    out[f"seed{seed}"] = {"both_gold": float(fa.both_gold(top).mean()),
                          "silos_searched": float(searched.sum(1).mean()),
                          "distance_computations": float((searched * sizes).sum(1).mean()),
                          "nn_recall_vs_central": float(nn)}
    print(seed, out[f"seed{seed}"], flush=True)
keys = ["both_gold", "silos_searched", "distance_computations", "nn_recall_vs_central"]
out["mean"] = {k: float(np.mean([out[f"seed{s}"][k] for s in fa.SEEDS])) for k in keys}
json.dump(out, open("results/routing_cost.json", "w"), indent=1)
print("mean", out["mean"])
