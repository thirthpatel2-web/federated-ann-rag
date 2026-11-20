"""Fair comparison for learned_routing.py: centroid routing at several temperatures traces a
cost-quality curve (higher temperature = budget spread over more silos). Results -> results/routing_frontier.json"""
import json

import faiss
import numpy as np

import federated_ann as fa

TEMPS = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2]
out = {}
for seed in fa.SEEDS:
    silos = fa.make_silos(10, "topic", seed)
    sizes = np.array([len(s) for s in silos])
    per_silo = []
    for ids in silos:
        index = faiss.IndexFlatIP(fa.d)
        index.add(fa.P[ids])
        sc, li = index.search(fa.Q, fa.K)
        per_silo.append((sc, ids[li]))
    for t in TEMPS:
        fa.TEMP = t
        b = fa.allocate(silos)
        top = fa.merge(per_silo, b)
        out.setdefault(str(t), []).append([float(fa.both_gold(top).mean()), float((b > 0).sum(1).mean()),
                                           float(((b > 0) * sizes).sum(1).mean())])
res = {t: dict(zip(["both_gold", "silos_searched", "distance_computations"], np.mean(v, 0).tolist())) for t, v in out.items()}
json.dump(res, open("results/routing_frontier.json", "w"), indent=1)
for t, r in res.items():
    print(t, {k: round(v, 3) for k, v in r.items()})
