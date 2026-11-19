"""Step 3: one chart from results/results.json."""
import json

import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))


def series(silos, index, param, mode):
    pts = [(r["S"], r["both_gold"]) for r in R
           if r["index"] == index and r["param"] == param and r["per_silo_k"] == mode
           and (r["silos"] == silos or r["S"] == 1)]
    return zip(*sorted(pts))


fig, ax = plt.subplots(figsize=(6.4, 3.8))
lines = [("topic", "flat", None, "full", "Exact search, every silo returns 10"),
         ("topic", "hnsw", 16, "full", "HNSW (efSearch=16), every silo returns 10"),
         ("random", "flat", None, "budget", "Exact search, random silos, 10/S per silo"),
         ("topic", "flat", None, "budget", "Exact search, topic silos, 10/S per silo")]
for silos, index, param, mode, label in lines:
    x, y = series(silos, index, param, mode)
    ax.plot(x, y, marker="o", label=label)
ax.set_xlabel("Number of silos (S)")
ax.set_ylabel("Questions with both gold passages in top-10")
ax.set_xticks([1, 2, 5, 10])
ax.set_ylim(0, 0.5)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, loc="lower left")
ax.set_title("HotpotQA, 1,000 questions, 66,581 passages", fontsize=9)
fig.tight_layout()
fig.savefig("results/both_gold_vs_silos.png", dpi=200)
print("saved results/both_gold_vs_silos.png")
