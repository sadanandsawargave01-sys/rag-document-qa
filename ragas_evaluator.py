"""
Ragas evaluation integration for EduBot RAG.

Ragas 0.4.3 is used as an offline/on-demand evaluator.
The evaluator is intentionally NOT called on every production request unless
RAGAS_AUTO_EVALUATE=true, because Ragas metrics make additional LLM calls.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def _result_value(result: Any) -> Optional[float]:
    """Normalize Ragas score objects/floats into a JSON-safe float."""
    if result is None:
        return None

    value = getattr(result, "value", result)

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class RagasEvaluator:
    """
    Evaluates one RAG interaction.

    Metrics:
      - Faithfulness: answer grounded in retrieved contexts.
      - Response relevancy: answer addresses the user question.
      - Context precision: relevant contexts are ranked highly (needs reference).
      - Context recall: retrieved contexts cover the reference answer (needs reference).
      - Answer correctness: answer correctness against reference (needs reference).
    """

    def __init__(
        self,
        google_api_key: str,
        evaluator_model: Optional[str] = None,
    ):
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is required for Ragas evaluation.")

        self.google_api_key = google_api_key
        self.evaluator_model = (
            evaluator_model
            or os.getenv("RAGAS_EVALUATOR_MODEL", "gemini-2.5-flash")
        )

        # Reuse the same Google Gemini provider already used by the application.
        self.llm = LangchainLLMWrapper(
            ChatGoogleGenerativeAI(
                model=self.evaluator_model,
                google_api_key=self.google_api_key,
                temperature=0,
                max_retries=2,
            )
        )

        # Used by relevancy/correctness metrics.
        self.embeddings = LangchainEmbeddingsWrapper(
            GoogleGenerativeAIEmbeddings(
                model=os.getenv(
                    "RAGAS_EMBEDDING_MODEL",
                    "models/gemini-embedding-001",
                ),
                google_api_key=self.google_api_key,
            )
        )

        self._build_metrics()

    def _build_metrics(self):
        # Ragas 0.4.x exposes these through ragas.metrics.
        from ragas.metrics import (
            Faithfulness,
            ContextPrecision,
            ContextRecall,
            AnswerCorrectness,
        )

        self.faithfulness = Faithfulness(llm=self.llm)

        # ResponseRelevancy is the current terminology. Keep a fallback for
        # installations where the metric is still named AnswerRelevancy.
        try:
            from ragas.metrics import ResponseRelevancy
            self.response_relevancy = ResponseRelevancy(
                llm=self.llm,
                embeddings=self.embeddings,
            )
        except ImportError:
            from ragas.metrics import AnswerRelevancy
            self.response_relevancy = AnswerRelevancy(
                llm=self.llm,
                embeddings=self.embeddings,
            )

        self.context_precision_cls = ContextPrecision
        self.context_recall_cls = ContextRecall
        self.answer_correctness_cls = AnswerCorrectness

    async def evaluate(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str],
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG interaction.

        Reference is optional:
        - without reference -> faithfulness + response relevancy
        - with reference -> also context precision, context recall,
          and answer correctness
        """
        contexts = [
            str(c).strip()
            for c in (retrieved_contexts or [])
            if c and str(c).strip()
        ]

        if not user_input or not response:
            raise ValueError("user_input and response are required.")

        if not contexts:
            # Ragas faithfulness needs retrieved_contexts.
            return {
                "status": "skipped",
                "reason": "No retrieved contexts were supplied.",
                "scores": {},
            }

        # Build one sample for all metrics.
        from ragas.dataset_schema import SingleTurnSample

        sample_kwargs = {
            "user_input": user_input,
            "response": response,
            "retrieved_contexts": contexts,
        }

        if reference and reference.strip():
            sample_kwargs["reference"] = reference.strip()

        sample = SingleTurnSample(**sample_kwargs)

        scores: Dict[str, Optional[float]] = {}
        errors: Dict[str, str] = {}

        async def run_metric(name: str, metric: Any):
            try:
                value = await metric.single_turn_ascore(sample)
                scores[name] = _result_value(value)
            except Exception as exc:
                scores[name] = None
                errors[name] = str(exc)

        # Core production-safe metrics.
        await run_metric("faithfulness", self.faithfulness)
        await run_metric("response_relevancy", self.response_relevancy)

        # Reference-based metrics are intentionally only enabled when a
        # ground-truth answer is supplied.
        if reference and reference.strip():
            context_precision = self.context_precision_cls(llm=self.llm)
            context_recall = self.context_recall_cls(llm=self.llm)
            answer_correctness = self.answer_correctness_cls(
                llm=self.llm,
                embeddings=self.embeddings,
            )

            await run_metric("context_precision", context_precision)
            await run_metric("context_recall", context_recall)
            await run_metric("answer_correctness", answer_correctness)

        available = [
            value for value in scores.values()
            if isinstance(value, (int, float))
        ]

        return {
            "status": "success",
            "evaluator_model": self.evaluator_model,
            "reference_used": bool(reference and reference.strip()),
            "context_count": len(contexts),
            "scores": scores,
            "errors": errors,
            "overall_score": (
                round(sum(available) / len(available), 4)
                if available else None
            ),
        }


async def evaluate_rag_response(
    google_api_key: str,
    user_input: str,
    response: str,
    docs: List[Dict[str, Any]],
    reference: Optional[str] = None,
    evaluator_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function used by FastAPI.
    """
    contexts = [
        doc.get("text", "")
        for doc in (docs or [])
        if doc.get("text")
    ]

    evaluator = RagasEvaluator(
        google_api_key=google_api_key,
        evaluator_model=evaluator_model,
    )

    result = await evaluator.evaluate(
        user_input=user_input,
        response=response,
        retrieved_contexts=contexts,
        reference=reference,
    )

    # Return source metadata so low-scoring retrieval can be diagnosed.
    result["retrieved_sources"] = [
        {
            "pdf_name": doc.get("pdf_name"),
            "page_no": doc.get("page_no"),
            "score": doc.get("score"),
            "rerank_score": doc.get("_rerank_score"),
        }
        for doc in (docs or [])
    ]

    return result
