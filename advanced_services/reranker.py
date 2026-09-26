import os
import re
from functools import lru_cache
import time
import os
os.environ["ENABLE_CROSS_ENCODER_RERANKER"] = "true"

print("LOADED RERANKER FROM:", __file__)

def _tokens(text: str) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in re.findall(r"[\w\u0900-\u097F]+", text.lower())
        if len(token) > 2
    }


class Reranker:
    """Optional cross-encoder reranker with a lightweight lexical fallback.

    Set ENABLE_CROSS_ENCODER_RERANKER=true to enable the heavier model. By
    default this keeps local startup fast and avoids extra downloads while still
    providing deterministic reranking.
    """

    def __init__(self) -> None:
        self.model = self._load_model()
        print("Cross Encoder:", self.model is not None)

        if self.model:
            print("Loaded reranker:", type(self.model).__name__)

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_model():
        print(
            "ENABLE_CROSS_ENCODER_RERANKER =",
            os.getenv("ENABLE_CROSS_ENCODER_RERANKER")
        )
        if os.getenv("ENABLE_CROSS_ENCODER_RERANKER", "").lower() not in {"1", "true", "yes"}:
            return None
        try:
            from sentence_transformers import CrossEncoder

            #return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            return CrossEncoder(
                "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
            )
        except Exception as exc:
            print("Cross-encoder reranker unavailable, using lexical fallback:", exc)
            return None

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:

        if not docs:
            return []    

        if self.model is not None:
            print("\nTOP BEFORE RERANK")
            for doc in docs[:5]:
                print(
                    f"vector={doc['score']:.4f}",
                    f"page={doc.get('page_no')}"
                )
            pairs = [[query, doc.get("text", "")] for doc in docs]
           
            print(f"Reranking {len(docs)} docs")
            print("RERANK START")
            print("DOCS BEFORE:", len(docs))
           
            start = time.time()
            scores = self.model.predict(pairs)
           
            for i, score in enumerate(scores[:10]):
                print(f"Rerank score {i}: {score}")
            print(f"Rerank time: {time.time() - start:.2f} sec")
            import math

            for doc, score in zip(docs, scores):
                doc["_vector_score"] = doc.get("score")
                probability_score = 1 / (1 + math.exp(-float(score)))
                doc["_rerank_score"] = probability_score
            #    doc["_rerank_score"] = float(score)
                
            # sorted_docs = sorted(
            #     docs,
            #     key=lambda x: x["_rerank_score"],
            #     reverse=True
            # )  
            print("TOP AFTER RERANK")
           
            for doc in sorted(
                docs,
                key=lambda x: x["_rerank_score"],
                reverse=True
            )[:5]:
                print(doc["_rerank_score"],doc.get("page_no"))
                print(
                    f"rerank={doc['_rerank_score']:.4f}",
                    f"vector={doc['_vector_score']:.4f}",
                    f"page={doc.get('page_no')}"
                )
            return sorted(docs, key=lambda doc: doc.get("_rerank_score", 0.0), reverse=True)

        query_tokens = _tokens(query)
        for doc in docs:
            doc_tokens = _tokens(f"{doc.get('pdf_name', '')} {doc.get('text', '')}")
            overlap = len(query_tokens & doc_tokens) / len(query_tokens) if query_tokens else 0.0
            base_score = float(doc.get("_rerank_score", 0.0))
            doc["_rerank_score"] = base_score + min(0.15, overlap * 0.15)

        return sorted(docs, key=lambda doc: doc.get("_rerank_score", 0.0), reverse=True)

