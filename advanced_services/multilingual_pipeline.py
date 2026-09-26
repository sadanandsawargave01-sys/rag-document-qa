from language_utils import detect_query_language


class MultilingualPipeline:
    """Central place for query language normalization.

    The embedding model is multilingual, so the original query should remain
    the primary search text. English translation can still be added as a
    secondary variant by redis_search.py for cross-language document matches.
    """

    def normalize(self, query: str) -> dict:
        language = detect_query_language(query)
        return {
            "original_query": query,
            "translated_query": query,
            "language": language,
        }

