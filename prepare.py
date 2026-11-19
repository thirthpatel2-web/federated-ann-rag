"""Step 1: build the passage corpus and the question set, then embed everything.

Input: data/hotpot_distractor_validation.parquet, downloaded from
https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet

Corpus: every paragraph that appears in the HotpotQA distractor validation set
(10 paragraphs per question, gold + distractors), de-duplicated by title.
Gold labels: the two "supporting" paragraph titles of each question.

Outputs (in data/):
  corpus.jsonl      one passage per line: {"pid", "title", "text"}
  questions.jsonl   sampled questions: {"qid", "question", "answer", "type", "gold_pids"}
  passages.npy      passage embeddings (float32, L2-normalised)
  queries.npy       question embeddings (same model)
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

DATA = Path("data")
N_QUESTIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small (384-dim), runs fine on a laptop CPU

rows = pq.read_table(DATA / "hotpot_distractor_validation.parquet").to_pylist()

# 1) corpus: one passage per unique Wikipedia title
title_to_pid, corpus = {}, []
for r in rows:
    for title, sents in zip(r["context"]["title"], r["context"]["sentences"]):
        if title not in title_to_pid:
            title_to_pid[title] = len(corpus)
            corpus.append({"pid": len(corpus), "title": title, "text": "".join(sents).strip()})

# 2) questions: random sample with a fixed seed so results are reproducible
random.seed(13)
sample = random.sample(rows, N_QUESTIONS)
questions = []
for i, r in enumerate(sample):
    gold = sorted({title_to_pid[t] for t in r["supporting_facts"]["title"]})
    questions.append({"qid": i, "question": r["question"], "answer": r["answer"],
                      "type": r["type"], "gold_pids": gold})

with open(DATA / "corpus.jsonl", "w", encoding="utf-8") as f:
    for p in corpus:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(DATA / "questions.jsonl", "w", encoding="utf-8") as f:
    for q in questions:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
print(f"corpus: {len(corpus)} passages | questions: {len(questions)}")

# 3) embeddings: normalised, so inner product == cosine similarity
model = SentenceTransformer(MODEL, device="cpu")
P = model.encode([f"{p['title']}. {p['text']}" for p in corpus], batch_size=128,
                 show_progress_bar=True, normalize_embeddings=True, convert_to_numpy=True)
Q = model.encode([q["question"] for q in questions], batch_size=128,
                 normalize_embeddings=True, convert_to_numpy=True)
np.save(DATA / "passages.npy", P.astype("float32"))
np.save(DATA / "queries.npy", Q.astype("float32"))
print("embeddings:", P.shape, Q.shape)
