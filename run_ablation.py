"""
Retrieval ablation: BM25 vs dense vs hybrid (RRF) vs hybrid + cross-encoder.

Runs batch_evaluate.py in retrieval-only mode (--no-llm) once per configuration
on the same held-out question set, then writes one comparison table:

    eval_results/ablation_retrieval.csv
    eval_results/ablation_retrieval.md

Every configuration shares the same index, query router, query expansion,
reference-section filter and evidence re-scoring; only the retrieval channels
and the reranker change, so differences isolate each stage's contribution.

Relevance is paper-level: a question is a hit at rank k if a chunk from the
question's source paper appears in the top-k returned chunks (the labelled set
gives the source paper, not a gold passage).

Usage:
    python run_ablation.py --dataset evaluation_set_100papers.json
    python run_ablation.py --dataset evaluation_set_100papers.json --max 50   # quick check
"""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

CONFIGS = [
    ("BM25 only",                          ["--retriever", "bm25",   "--no-reranker"]),
    ("Dense only (FAISS + BGE-M3)",        ["--retriever", "dense",  "--no-reranker"]),
    ("Hybrid (Dense + Sparse + BM25, RRF)", ["--retriever", "hybrid", "--no-reranker"]),
    ("Hybrid + cross-encoder reranker",    ["--retriever", "hybrid"]),
]

COLUMNS = ["hit_rate", "mrr", "recall@1", "recall@3", "recall@5", "recall@10"]


def latest_json(folder: Path) -> str:
    files = sorted(glob.glob(str(folder / "*.json")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(f"no result json in {folder}")
    return files[-1]


def main():
    parser = argparse.ArgumentParser(description="Retrieval ablation runner")
    parser.add_argument("--dataset", "-d", default="evaluation_set_100papers.json")
    parser.add_argument("--max", "-n", type=int, default=None)
    parser.add_argument("--out", default="eval_results")
    args = parser.parse_args()

    out_dir = BASE_DIR / args.out
    out_dir.mkdir(exist_ok=True)
    rows = []

    for name, flags in CONFIGS:
        slug = flags[1] + ("_rerank" if "--no-reranker" not in flags else "")
        run_dir = out_dir / "ablation_runs" / slug
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(BASE_DIR / "batch_evaluate.py"),
               "--dataset", args.dataset, "--output", str(run_dir), "--no-llm", *flags]
        if args.max:
            cmd += ["--max", str(args.max)]
        print("\n" + "=" * 70 + f"\n{name}\n  " + " ".join(cmd) + "\n" + "=" * 70)
        ret = subprocess.run(cmd, cwd=str(BASE_DIR))
        if ret.returncode != 0:
            print(f"[ERROR] {name} exited with {ret.returncode}")
            continue
        with open(latest_json(run_dir), encoding="utf-8") as f:
            summary = json.load(f)["summary"]
        r = summary["retrieval"]
        rows.append({
            "configuration": name,
            "questions": summary["completed"],
            **{c: round(r[c], 4) for c in COLUMNS},
            "sec_per_question": summary.get("avg_time_per_question"),
        })

    with open(out_dir / "ablation_retrieval.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["| Configuration | Questions | Hit@10 | MRR@10 | R@1 | R@3 | R@5 | R@10 | s/question |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['configuration']} | {row['questions']} | {row['hit_rate']:.3f} | {row['mrr']:.3f} | "
            f"{row['recall@1']:.3f} | {row['recall@3']:.3f} | {row['recall@5']:.3f} | "
            f"{row['recall@10']:.3f} | {row['sec_per_question']} |"
        )
    (out_dir / "ablation_retrieval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
