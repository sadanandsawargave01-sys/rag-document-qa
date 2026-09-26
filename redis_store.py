
import asyncio
import logging
import re
from functools import partial
from language_utils import detect_query_language

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Basic text cleaning for better embeddings"""
    text = re.sub(r'\s+', ' ', text)  # remove extra spaces
    return text.strip()

def detect_lang_safe(text: str) -> str:
    """Safe language detection"""
    return detect_query_language(text, default="unknown")

async def store_pdf_chunks_in_redis(
    index_name: str,
    pdf_file_name: str,
    chunks: list,
    redis_client,
    embedding_model,        # SentenceTransformer instance
    s3_location: str,
    user_id: str | int,
    category_id: str | int,
    file_uuid: str,
    client_id: str,
    batch_size: int = 128    # 🔥 batch processing
) -> dict:
    """
    Multilingual optimized Redis storage with batching + pipeline.
    """

    index_full_name = f"{client_id}_{index_name}"
    loop = asyncio.get_event_loop()

    stored, skipped, failed = 0, 0, 0

    # 🔥 Step 1: Prepare clean texts
    processed_chunks = []
    metadata_list = []

    for i, chunk in enumerate(chunks):
        if i == 0:
            print("=" * 100)
            print("REDIS STORE DEBUG")
            print("INDEX_NAME:", index_name)
            print("CATEGORY_ID:", category_id)
            print("USER_ID:", user_id)
            print("PDF_NAME:", pdf_file_name)
            print("FILE_UUID:", file_uuid)
            print("PAGE_NO:", page_no)
            print("TEXT:")
            print(chunk[:500])
            print("=" * 100)
        try:
            text = chunk if isinstance(chunk, str) else chunk.get("text", "")

            if not text or not text.strip():
                skipped += 1
                continue

            text = clean_text(text)

            processed_chunks.append(text)
            metadata_list.append((i, text))

        except Exception as e:
            failed += 1
            logger.error(f"Preprocessing error chunk {i}: {e}")

    whole_text = " ".join(processed_chunks[:20])
    document_language = detect_lang_safe(whole_text)
    PIPELINE_SIZE = 1500
    doc_prefix = f"{index_full_name}:{file_uuid}:"
    # 🔥 Step 2: Batch embedding
    for start in range(0, len(processed_chunks), batch_size):
        batch_texts = processed_chunks[start:start + batch_size]
        batch_meta = metadata_list[start:start + batch_size]
        try:
            embeddings = await loop.run_in_executor(
                None,
#                partial(embedding_model.encode, batch_texts)
                lambda:embedding_model.encode(
                    batch_texts,
                    batch_size=128,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            )

            # 🔥 Step 3: Redis pipeline (fast writes)
            pipe = redis_client.pipeline()
            count = 0
            records = []
            for (i, text), emb in zip(batch_meta, embeddings):
                try:
#                    embedding_bytes = emb.astype("float32").tobytes()
                    embedding_bytes = emb.astype(np.float32, copy=False).tobytes()
#                    doc_id = f"{index_full_name}:{file_uuid}:{i}"
                    doc_id = doc_prefix + str(i)
                    # 🌐 detect language (optional but useful)
#                    lang = detect_lang_safe(text)
#                    pipe.hset(
#                        doc_id,
#                        mapping={
                    pipe.hset(
                        doc_id,
                        mapping={
                            "text": text,
                            "pdf_name": pdf_file_name,
                            "url": s3_location,
                            "chunk_id": str(i),
                            "user_id": str(user_id),
                            "category_id": str(category_id),
                            "embedding": embedding_bytes,
                            "language": document_language,   # ✅ NEW FIELD
                        }
                    )
                    count += 1
                    stored += 1
                    if count >= PIPELINE_SIZE:
                        await pipe.execute()
                        pipe = redis_client.pipeline()
                        count = 0

                except Exception as e:
                    failed += 1
                    logger.error(f"Chunk store error {i}: {e}")
            if count:
                await pipe.execute()
#            await pipe.execute()

        except Exception as e:
            failed += len(batch_texts)
            logger.error(f"Batch embedding error: {e}")

    logger.info(
        f"Redis store complete for '{pdf_file_name}': "
        f"stored={stored}, skipped={skipped}, failed={failed}"
    )

    return {
        "stored": stored,
        "skipped": skipped,
        "failed": failed
    }
