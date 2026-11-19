"""Step 3: two charts from results/results.json."""
import json

import matplotlib.pyplot as plt

R = json.load(open("results/results.json"))


def pick(**kw):
    return [r for r in R if all(r[k] == v for k, v in kw.items())]


fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))

# (a) result budgets with exact search: how much does routing recover?
central = pick(silos="central", index="flat")[0]["both_gold"]
for silos, style in [("topic", "-"), ("random", "--")]:
    for budget, label in [("equal", "equal split"), ("routed", "routed by centroid")]:
        rows = sorted(pick(silos=silos, index="flat", budget=budget), key=lambda r: r["S"])
        a1.plot([1] + [r["S"] for r in rows], [central] + [r["both_gold"] for r in rows],
                style, marker="o", label=f"{silos} silos, {label}")
a1.set_xlabel("Number of silos (S)")
a1.set_ylabel("Questions with both gold paragraphs in top-10")
a1.set_xticks([1, 2, 5, 10])
a1.set_ylim(0, 0.5)
a1.set_title("(a) Exact search, total budget of 10 results", fontsize=9)
a1.legend(fontsize=7)
a1.grid(alpha=0.3)

# (b) HNSW at matched search effort: central vs 10 silos
for silos, S, label in [("central", 1, "central index"), ("topic", 10, "10 topic silos"),
                        ("random", 10, "10 random silos")]:
    rows = sorted(pick(silos=silos, S=S, index="hnsw", budget="full"), key=lambda r: r["work_per_q"])
    a2.plot([r["work_per_q"] for r in rows], [r["both_gold"] for r in rows], marker="o", label=label)
    for r in rows:
        a2.annotate(f"ef={r['param']}", (r["work_per_q"], r["both_gold"]), fontsize=6,
                    xytext=(3, -9), textcoords="offset points")
a2.axhline(central, color="grey", lw=0.8, ls=":", label="exact search")
a2.set_xscale("log")
a2.set_xlabel("Distance computations per query (all silos)")
a2.set_ylabel("Both gold in top-10")
a2.set_title("(b) HNSW, every silo returns 10", fontsize=9)
a2.legend(fontsize=7)
a2.grid(alpha=0.3)

fig.suptitle("HotpotQA: 1,000 questions, 66,581 paragraphs, mean of 3 seeds", fontsize=9)
fig.tight_layout()
fig.savefig("results/federation_results.png", dpi=200)
print("saved results/federation_results.png")
