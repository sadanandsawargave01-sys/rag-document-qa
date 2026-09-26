import re


class QueryDecomposer:
    """Small deterministic fallback when LLM sub-query generation is empty."""

    def generate_sub_queries(self, query: str) -> list[str]:
        if not query or not query.strip():
            return []

        parts = re.split(
            r"\b(?:and|vs|versus|compare|difference between)\b|[,;]",
            query,
            flags=re.IGNORECASE,
        )

        results = []
        for part in parts:
            part = part.strip()
            if len(part) > 3 and part not in results:
                results.append(part)

        return results or [query.strip()]

