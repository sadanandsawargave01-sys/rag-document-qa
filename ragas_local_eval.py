"""
Run Ragas locally against the same Redis + RedisSearch RAG pipeline.

Usage:
    python ragas_local_eval.py
    python ragas_local_eval.py --dataset ragas_eval_dataset.jsonl
    python ragas_local_eval.py --dataset my_eval.jsonl --output ragas_results.json

Dataset JSONL fields:
{
  "user_query": "...",
  "index_name": "...",
  "client_id": "...",
  "user_id": null,
  "category_id": null,
  "reference_answer": "..."
}

reference_answer is optional. If supplied, Ragas also computes:
  context_precision, context_recall, answer_correctness.
Without it, it computes:
  faithfulness, response_relevancy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from config import GOOGLE_API_KEY, REDIS_HOST, REDIS_PORT
from redis_search import RedisSearch
from ragas_evaluator import RagasEvaluator
from language_utils import detect_query_language, normalize_language_code
import redis.asyncio as redis


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no}: {exc}"
                ) from exc
    return rows


async def evaluate_case(
    case: Dict[str, Any],
    redis_client,
    ragas_evaluator: RagasEvaluator,
) -> Dict[str, Any]:
    user_query = case["user_query"]
    index_name = case["index_name"]
    client_id = case["client_id"]
    user_id = case.get("user_id")
    category_id = case.get("category_id")
    reference = case.get("reference_answer")

    detected_lang = normalize_language_code(
        detect_query_language(user_query)
    )

    redis_search = RedisSearch(
        redis_client,
        GOOGLE_API_KEY,
    )

    analysis = await redis_search.unified_query_analysis(
        "",
        user_query,
    )

    answer_type = analysis.get("answer_type", "other")
    query_tag = analysis.get("query_tag", "Exact")
    query = analysis.get("reframed_question", user_query)
    sub_queries = analysis.get("sub_queries", [])

    full_index_name = f"{client_id}_{index_name}"

    if query_tag == "Exact":
        answer, docs = await redis_search.query_search(
            query,
            user_id,
            category_id,
            client_id,
            full_index_name,
            answer_type,
            target_language=detected_lang,
        )
    else:
        answer, docs = await redis_search.research_query_search_with_subqueries(
            query,
            user_id,
            category_id,
            client_id,
            full_index_name,
            answer_type,
            sub_queries,
            target_language=detected_lang,
        )

    evaluation = await ragas_evaluator.evaluate(
        user_input=user_query,
        response=answer or "",
        retrieved_contexts=[
            d.get("text", "")
            for d in (docs or [])
            if d.get("text")
        ],
        reference=reference,
    )

    return {
        "user_query": user_query,
        "reframed_question": query,
        "query_tag": query_tag,
        "answer_type": answer_type,
        "answer": answer,
        "retrieved_doc_count": len(docs or []),
        "evaluation": evaluation,
        "sources": [
            {
                "pdf_name": d.get("pdf_name"),
                "page_no": d.get("page_no"),
                "score": d.get("score"),
                "rerank_score": d.get("_rerank_score"),
            }
            for d in (docs or [])
        ],
    }


async def main(dataset_path: str, output_path: str):
    cases = load_jsonl(dataset_path)

    if not cases:
        raise ValueError("Evaluation dataset is empty.")

    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=False,
    )

    evaluator = RagasEvaluator(
        google_api_key=GOOGLE_API_KEY,
        evaluator_model=os.getenv(
            "RAGAS_EVALUATOR_MODEL",
            "gemini-2.5-flash",
        ),
    )

    results = []

    for i, case in enumerate(cases, 1):
        print("\n" + "=" * 100)
        print(f"RAGAS TEST CASE {i}/{len(cases)}")
        print("QUESTION:", case.get("user_query"))
        print("=" * 100)

        try:
            result = await evaluate_case(
                case,
                redis_client,
                evaluator,
            )
            results.append(result)

            scores = result["evaluation"].get("scores", {})
            print("ANSWER:", result["answer"])
            print("SCORES:")
            for name, score in scores.items():
                print(f"  {name:22s}: {score}")
            print(
                "OVERALL:",
                result["evaluation"].get("overall_score"),
            )

        except Exception as exc:
            print("FAILED:", exc)
            results.append({
                "user_query": case.get("user_query"),
                "status": "failed",
                "error": str(exc),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("RAGAS EVALUATION COMPLETE")
    print("RESULT FILE:", output_path)
    print("=" * 100)

    await redis_client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="ragas_eval_dataset.jsonl",
        help="JSONL evaluation dataset",
    )
    parser.add_argument(
        "--output",
        default="ragas_results.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    asyncio.run(
        main(
            dataset_path=args.dataset,
            output_path=args.output,
        )
    )
