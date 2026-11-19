# What does it cost to search data that cannot be merged?

A small experiment, still in progress. I built it while preparing a research internship
application, because of a problem I hit in my own project
[CivicLens](https://github.com/thirthpatel2-web/Civiclens): government departments can only
share data with consent, so their documents cannot go into one central search index.

**Question:** if the documents are split across separate "silos", each with its own vector
index, how much retrieval quality do we lose, and where does the loss come from?

## Setup

- **Data:** the [HotpotQA](https://hotpotqa.github.io/) distractor validation set. Every
  paragraph in it (66,581 unique Wikipedia paragraphs) forms the corpus; 1,000 questions are
  sampled with a fixed seed. Each question has two gold "supporting" paragraphs.
- **Embeddings:** `all-MiniLM-L6-v2` (384 dimensions), the same model for every silo.
- **Silos:** 1 (central), 2, 5 or 10 silos, built either by topic (k-means on the embeddings)
  or at random.
- **Indexes (Faiss):** Flat (exact), HNSW (M=32, efSearch 16 or 64), IVF-PQ
  (nlist = sqrt(n), 16 sub-quantizers of 8 bits, nprobe 1 or 8).
- **Federated search:** query every silo, merge all candidates by score, keep the top 10.
  Each silo returns either 10 results ("full") or ceil(10/S) ("budget").
- **Metrics:** overlap with the exact central top-10, share of gold paragraphs found, and
  share of questions with *both* gold paragraphs in the top-10 (multi-hop questions need both).

## Results so far (1,000 questions, k = 10)

![results](results/both_gold_vs_silos.png)

| Setting | Both gold in top-10 |
|---|---|
| Central exact index | 46.0% |
| Any number of silos, exact search, every silo returns 10 | 46.0% (identical to central) |
| Central HNSW, efSearch=16 | 38.7% |
| 10 topic silos, HNSW efSearch=16, every silo returns 10 | 41.5% |
| 10 random silos, exact, 1 result per silo | 35.8% |
| 10 topic silos, exact, 1 result per silo | 13.8% |
| Central IVF-PQ, nprobe=8 (16 x 8-bit codes) | 16.3% |

What I take from this so far:

1. **Splitting alone costs nothing.** With exact search, full results from every silo and
   merging by score (one shared embedding model), the merged top-10 is identical to the
   central top-10, as expected.
2. **Per-silo HNSW looked better, but only because it does more work.** With 10 silos each
   query searches 10 graphs (38.7% central vs 41.5% topic / 44.2% random silos at
   efSearch=16). At comparable work a central HNSW with efSearch=64 reaches 44.0%, so this is
   not a free gain. I am re-running the comparison at matched search effort.
3. **Fixed per-silo result budgets are the big loss for topic silos.** When each silo may
   return only ceil(10/S) results, topic silos drop from 46.0% to 13.8% at 10 silos. This is
   largely structural: in 55% of questions both gold paragraphs are in the same topic silo
   (80% at 2 silos, 65% at 5), so with one result per silo at most 45% of questions can ever
   get both. Random silos lose much less (35.8%). The interesting question is how to share a
   fixed total budget across silos.
4. **IVF-PQ numbers are not a fair comparison yet.** Each silo trains its own PQ codebooks,
   so their approximate scores are not comparable when merged, and there is no re-ranking.
   I am fixing this (exact re-ranking of candidates, plus an IVF-Flat baseline).

## Limitations and next steps

- Retrieval only. Next: generate answers with a small open model and check whether these
  retrieval losses change the final answer.
- One embedding model shared by all silos (optimistic), one dataset, one run per setting.
- The corpus is HotpotQA's distractor paragraphs, which were chosen to look like the gold ones,
  so it is harder and smaller than a real corpus.
- Timings were measured on a laptop CPU while other jobs were running, so I do not report them.
- Next: smarter budgets (ask more results from the silos that look most relevant) and
  choosing silos with a language model.

## Run it

```
pip install faiss-cpu sentence-transformers pyarrow matplotlib
# download HotpotQA distractor validation parquet into data/ (see prepare.py)
python prepare.py 1000      # corpus, questions, embeddings (~40 min on a laptop CPU)
python federated_ann.py     # all settings -> results/results.json
python plot.py              # chart
```
