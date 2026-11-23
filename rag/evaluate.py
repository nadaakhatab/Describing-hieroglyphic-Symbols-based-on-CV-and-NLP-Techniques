"""Offline retrieval evaluation for Precision@K, Recall@K, MRR and nDCG.

Input JSONL format, one query per line:
{"query":"What is the ankh?","relevant_ids":["sign-S34"]}

Run from the project folder:
    python -m rag.evaluate --dataset rag/data/evaluation.jsonl --k 1 3 5
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rag.retrieval import retrieve


def evaluate(dataset: list[dict], ks: list[int]) -> dict[str, float]:
    results: dict[str, float] = {}
    for k in ks:
        precision, recall, hits, ndcg = [], [], [], []
        reciprocal_ranks = []
        for item in dataset:
            relevant = set(item["relevant_ids"])
            returned = [record["id"] for record in retrieve(item["query"], item.get("codes", []), top_k=k)]
            found = [doc_id in relevant for doc_id in returned]
            precision.append(sum(found) / k)
            recall.append(sum(found) / max(1, len(relevant)))
            hits.append(float(any(found)))
            rank = next((index + 1 for index, value in enumerate(found) if value), None)
            reciprocal_ranks.append(1 / rank if rank else 0.0)
            dcg = sum((1 / math.log2(index + 2)) for index, value in enumerate(found) if value)
            ideal = sum(1 / math.log2(index + 2) for index in range(min(k, len(relevant))))
            ndcg.append(dcg / ideal if ideal else 0.0)
        prefix = f"@{k}"
        results[f"precision{prefix}"] = sum(precision) / len(precision)
        results[f"recall{prefix}"] = sum(recall) / len(recall)
        results[f"hit_rate{prefix}"] = sum(hits) / len(hits)
        results[f"mrr{prefix}"] = sum(reciprocal_ranks) / len(reciprocal_ranks)
        results[f"ndcg{prefix}"] = sum(ndcg) / len(ndcg)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("rag/data/evaluation.jsonl"))
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(json.dumps({"queries": len(rows), **evaluate(rows, args.k)}, indent=2))


if __name__ == "__main__":
    main()
