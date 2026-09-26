try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except Exception:
        RecursiveCharacterTextSplitter = None


class SemanticChunker:
    """Reusable chunking helper for future ingestion changes.

    Existing QA bot ingestion is preserved, but this helper is included so new
    ingestion paths can use consistent multilingual chunk sizing.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "। ", " ", ""],
            )
            if RecursiveCharacterTextSplitter
            else None
        )

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []

        if self.splitter is not None:
            chunks = self.splitter.split_text(text)
        else:
            chunks = [
                text[index : index + self.chunk_size]
                for index in range(0, len(text), self.chunk_size - self.chunk_overlap)
            ]

        return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
