
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, HttpUrl, Field, ConfigDict
import redis.asyncio as redis
import asyncio
import os
import traceback
import logging
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
import time
import re
from advanced_services.semantic_chunking import SemanticChunker
from sqlalchemy.ext.asyncio import AsyncSession

# Internal Imports
from database import SessionLocal, get_db
from models import QABOTParser, Base
from s3 import upload_file_to_s3, get_presigned_url
from config import GOOGLE_API_KEY, REDIS_HOST, REDIS_PORT
from pdf_processing import PDFProcessor
from utils import get_file_type, save_upload_file_tmp, read_and_prepare_file
from redis_search import RedisSearch, embedding_dim, embedding_model as imported_model
from question_paper_generator import llm_question_paper_generator
from question_extraction import image_question_extraction
from llm_helpers import get_llm_general_answers, get_llm_summary
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE
from redis_search import detect_lang
from language_utils import detect_query_language, get_fallback_message, normalize_language_code
from semantic_cache import semantic_cache_search, store_semantic_cache
from ragas_evaluator import evaluate_rag_response, RagasEvaluator
import uvloop
import asyncio

asyncio.set_event_loop_policy(
    uvloop.EventLoopPolicy()
)


# --- Helpers ---
async def get_llm_summary_multilingual(content: str, target_language: str = 'en'):
    prompt = f"Please summarize the following content in {target_language} language:\n{content}"
    return await get_llm_summary(prompt)

def extract_urls_from_docs(docs):
    urls = []
    for d in docs:
        if isinstance(d, dict) and d.get("url"):
            page = d.get("page_no")
            url = d["url"]
            # âœ… attach page number for direct navigation
            if page:
                url = f"{url}#page={page}"

            urls.append(url)

#    return list(set(urls))
    seen = set()
    ordered_urls = []

    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered_urls.append(url)

    return ordered_urls


async def image_question_extraction_multilingual(image_bytes, target_language='en'):
    return image_question_extraction(image_bytes, target_language)

async def generate_question_paper_multilingual(knowledge_files, qp_format_file, target_language='en'):
    knowledge_paths = [save_upload_file_tmp(file) for file in knowledge_files]
    format_path = save_upload_file_tmp(qp_format_file)
    status, output_file_path = llm_question_paper_generator(
        knowledge_ip_image_path_list=[p for p in knowledge_paths if not p.endswith(".pdf")],
        knowledge_ip_pdf_path=next((p for p in knowledge_paths if p.endswith(".pdf")), ''),
        qp_format_pdf=format_path if format_path.endswith(".pdf") else '',
        qp_format_image=format_path if not format_path.endswith(".pdf") else '',
        target_language=target_language
    )
    return status, output_file_path

def clean_final_answer(text):
    import re
    if not text:
        return ""
    text = re.sub(r'<a.*?>.*?</a>', '', text, flags=re.DOTALL)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'Internal Knowledge Center Answer', '', text, flags=re.I)
    text = re.sub(r'DOCBrains Knowledge Center Answer', '', text, flags=re.I)
    text = re.sub(r'अंतर्गत ज्ञान केंद्र उत्तर', '', text, flags=re.I)
    text = re.sub(r'DOCBrains ज्ञान केंद्र उत्तर', '', text, flags=re.I)
    text = re.sub(r"Here's.*?:", "", text, flags=re.I) 
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    sentences = re.split(r'[.!?à¥¤]+\s*', text)
    unique_sentences = []
    seen = set()

    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean and s_clean not in seen:
            seen.add(s_clean)
            unique_sentences.append(s.strip())
    text = ". ".join(unique_sentences)  
    text = text.replace("How it works", "\n\nðŸ”¹ How it works:\n")
    text = text.replace("Examples of Generative AI", "\n\nðŸ”¹ Examples:\n")
    text = text.replace("Use Cases", "\n\nðŸ”¹ Use Cases:\n")
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_fallback_message(lang):
    from language_utils import get_fallback_message as _localized_fallback

    return _localized_fallback(lang)

def is_fallback_response(text: str):
    if not text or not text.strip():
        return True

    text_lower = text.strip().lower()
    localized_fallbacks = [
        get_fallback_message("en").lower(),
        get_fallback_message("hi").lower(),
        get_fallback_message("mr").lower(),
    ]
    if any(message in text_lower for message in localized_fallbacks):
        return True
    
    fallback_phrases = [
        "क्षम करा",
        "माहिती आढळली नाही",
        "क्षमा करें",
        "जानकारी नहीं मिली",
        "sorry",
        "could not find",
        "answer is not present in the system",
        "not present in the system",
        "no answer",
        "no data",
        "not available",
        "not available in the system",
        "not found in the",
        "not found",
        "not found in the system",
        "unable to find",
        "आपके प्रश्न का उत्तर सिस्टम में मौजूद नहीं है",
        "तुमच्या प्रश्नाचे उत्तर सिस्टममध्ये उपलब्ध नाही",
        "आपके सवाल का जवाब सिस्टम में मौजूद नहीं है",
        "तुमच्या प्रश्नाचे उत्तर प्रणालीमध्ये उपलब्ध नाही",
        "क्षमा करा",
        "तुमच्या प्रश्नाचे उत्तर सिस्टीममध्ये उपलब्ध नाही",
        "तुमच्या प्रश्नाचे उत्तर सिस्टममध्ये नाही",
        "आपल्या प्रश्नाचे उत्तर प्रणालीमध्ये उपलब्ध नाही",
        "For your question, the answer is not present in the system",
        "तुमच्या प्रश्नाचे उत्तर प्रणालीमध्ये उपलब्ध नाही.",
        "प्रदान केलेल्या दस्तऐवजांमध्ये",
        "मला प्रदान केलेल्या दस्तऐवजांमध्ये",
        "याबद्दल माहिती आढळली नाही",
    ]

    return any(p in text_lower for p in fallback_phrases)

def normalize_lang(lang):
    return normalize_language_code(lang)

# -----------------------------
# ðŸ”¥ NORMALIZE QUERY (CACHE FIX)
# -----------------------------
def normalize_query(q: str):
    q = q.lower().strip()
    q = q.replace("?", "")
    q = q.replace("।", "")
    q = re.sub(r"\s+", " ", q)      # remove extra spaces
    return q


# --- Logging ---
logger = logging.getLogger("uvicorn.error")
print(" RUNNING MAIN.PY")

app = FastAPI(title="EduBot RAG API")


# Initialize Redis
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)

# --- Global Model ---
embedding_model_instance = None

@app.on_event("startup")
async def startup_event():
    global embedding_model_instance
    logger.info("Checking embedding model status...")
    if hasattr(imported_model, 'encode'):
        embedding_model_instance = imported_model
    elif callable(imported_model):
        try:
            embedding_model_instance = imported_model()
        except Exception:
            embedding_model_instance = imported_model
    else:
        embedding_model_instance = imported_model

       # Initialize DB
    Base.metadata.create_all(bind=SessionLocal().bind)



# --- Models ---
class FileDetail(BaseModel):

    model_config = ConfigDict(from_attributes=True)
    FileUUID: str
    FileName: str
    Status: str
    S3Location: Optional[str] = None
    CreatedOn: Optional[datetime] = None
    CategoryId: str
    FileType: str
    FileSize: int
    IndexName: str

class QueryRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")
    user_query: Optional[str] = Field(default=None, alias="question")
    session_id: Optional[str] = Field(default="default_session")
    index_name: str
    user_id: Optional[int] = None
    category_id: Optional[str] = None
    client_id: str
    answer_type: Optional[str] = None 
    master_client: Optional[str] = None 
#    target_language: Optional[str] = Field(default='en', exclude=True)

class S3Input(BaseModel):
    s3_url: HttpUrl
    expiration: int = 3600
    disposition: str = "inline"

class S3Output(BaseModel):
    presigned_url: str


class RagasEvaluateRequest(BaseModel):
    """Evaluate a question against the same RAG pipeline used by /query/stream/."""
    user_query: str = Field(..., min_length=1)
    index_name: str
    client_id: str
    user_id: Optional[int] = None
    category_id: Optional[str] = None
    reference_answer: Optional[str] = None
    answer_type: Optional[str] = None
    master_client: Optional[str] = None
    evaluator_model: Optional[str] = None


class RagasResponseRequest(BaseModel):
    """Evaluate an already generated response and retrieved contexts."""
    user_query: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    retrieved_contexts: List[str] = Field(default_factory=list)
    reference_answer: Optional[str] = None
    evaluator_model: Optional[str] = None

# --- Helpers ---

async def create_or_update_index(index_name):
    try:
        logger.info(
            f"Creating index {index_name} with embedding_dim={embedding_dim}"
        )

        from redis.commands.search.field import VectorField, TextField, TagField
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        
        schema = [
            TextField("pdf_name"),
            TextField("page_no"),
            TagField("file_uuid"),
            VectorField("embedding", "HNSW", {
                "TYPE": "FLOAT32",
                "DIM": embedding_dim,
                "DISTANCE_METRIC": "COSINE"
            }),
            TextField("text"),
            TextField("chunk_id"),
            TextField("index_name"),
            TextField("url"),
            TextField("user_id"),
            TextField("category_id")
        ]
        # Prefix ensures the index only looks at keys starting with 'index_name:'
        index_def = IndexDefinition(prefix=[f'{index_name}:'], index_type=IndexType.HASH)
        await redis_client.ft(index_name).create_index(schema, definition=index_def)
        logger.info(f" Redis Index Verified: {index_name}")
    except Exception as e:
        # Index already exists or other non-critical error
        pass
#        logger.error(
#            f"Index creation failed for {index_name}: {str(e)}"
#        )

async def store_pdf_chunks_in_redis(index_name, pdf_file_name, chunks, redis_client, 
                                   s3_location, user_id, category_id, file_uuid):
    try:
        global embedding_model_instance
        loop = asyncio.get_running_loop()
        
        # Use the global instance
        model = embedding_model_instance or imported_model
        
        # Filter out empty chunks
        valid_chunks = [c for c in chunks if c and str(c).strip()]
        logger.info(
            f"STORE_CHUNKS | index={index_name} "
            f"| chunks={len(valid_chunks)} "
            f"| embedding_dim={embedding_dim}"
        )

        logger.info(
            f"Storing {len(valid_chunks)} chunks for {pdf_file_name}"
        )

#        for i, chunk in enumerate(valid_chunks[:5]):
#            logger.info(
#                f"Chunk {i+1}: {len(chunk)} chars"
#            )
        print("=" * 80)
        print("FIRST 10 CHUNKS STORED")

        for c in valid_chunks[:10]:
            print({
                "page_no": c["page_no"],
                "text_len": len(c["text"]),
                "text_preview": c["text"][:100]
            })
        print("=" * 80)
        if not valid_chunks:
            logger.warning(f" No valid text chunks found for {pdf_file_name}. Skipping storage.")
            return

#        for i, chunk in enumerate(valid_chunks):
        for i, chunk_data in enumerate(valid_chunks):
            chunk = chunk_data["text"]
            page_no = chunk_data["page_no"]
            print(
                f"STORING | chunk_id={i+1} | page_no={page_no} | text_len={len(chunk)}"
            )

            # Handle both model.encode() and direct call patterns
            if hasattr(model, 'encode'):
                embedding_func = lambda: model.encode(f"passage: {chunk}",normalize_embeddings=True).astype('float32').tobytes()
            else:
                embedding_func = lambda: model(chunk).astype('float32').tobytes()

            embedding= await loop.run_in_executor(None, embedding_func)
            if i == 0:
                logger.info(
                    f"EMBEDDING BYTE LENGTH={len(embedding)}"
                )


            # Key format: {full_index_name}:{file_uuid}:chunk{i}
            doc_id = f"{index_name}:{file_uuid}:chunk{i}"
            
            await redis_client.hset(doc_id, mapping={
                "pdf_name": pdf_file_name,
                "file_uuid": file_uuid,
                "chunk_id": str(i+1),
                "page_no": str(page_no),
                "text": chunk,
                "embedding": embedding,
                "index_name": index_name,
                "url": s3_location,
                "user_id": str(user_id),
                "category_id": str(category_id)
            })
        print(
            f"INDEX={index_name} "
            f"CATEGORY={category_id} "
            f"PDF={pdf_file_name}"
        )
        logger.info(f"Stored {len(valid_chunks)} chunks in {index_name}")
    except Exception as e:
        logger.error(f"Error storing chunks: {e}")
        traceback.print_exc()

# --- Diagnostic ---
@app.get("/debug/redis/{index_name}")
async def debug_redis(index_name: str):
    try:
        all_indexes = await redis_client.execute_command("FT._LIST")
        index_exists = index_name in [i.decode() if isinstance(i, bytes) else i for i in all_indexes]
        keys = await redis_client.keys(f"{index_name}:*")
        sample_data = []
        for key in keys[:3]:
            data = await redis_client.hgetall(key)
            readable = {k.decode(): (v.decode() if k.decode() != 'embedding' else f"<{len(v)} bytes>") for k, v in data.items()}
            sample_data.append({"key": key.decode(), "data": readable})
        return {"index": index_name, "exists": index_exists, "total_keys": len(keys), "samples": sample_data}
    except Exception as e:
        return {"error": str(e)}

# --- API Endpoints ---
@app.get("/test-working")
def test():
    return {"message": "NEW CODE RUNNING"}

@app.post("/index_data/")
async def index_data(index_name: str = Form(...), 
                    client_id: int = Form(...),
                    user_id: int = Form(...),
                    category_id: str = Form(...),
                    file: UploadFile = File(...),
                    db: Session = Depends(get_db)):
    try:
         full_index_name = f"{client_id}_{index_name}"
         logger.info(f" Indexing Request: {full_index_name}")
        
         await create_or_update_index(full_index_name)
        
         tmp_dir = "tmp_index"
         os.makedirs(tmp_dir, exist_ok=True)
        
         file_infos = read_and_prepare_file(file, tmp_dir)
         processed_any = False
        
         for info in file_infos:
             try:
                 s3_loc = upload_file_to_s3(info['file_bytes'], info['file_uuid'] + info['ext'])
                 processor = PDFProcessor(info['local_file_path'], info['file_uuid'])
                
                 # Extract text
                # chunks = processor.process_pdf() if info['ext'] == ".pdf" else [processor.process_image()]
                 page_texts = (
                    processor.process_pdf()
                    if info['ext'] == ".pdf"
                    else processor.process_image()
                 )

                 chunker = SemanticChunker(
                    chunk_size=1000,
                    chunk_overlap=200
                 )

                 chunks = []

                 for page_no, page_text in enumerate(page_texts, start=1):
                     page_chunks = chunker.chunk(page_text)

                     for chunk in page_chunks:
                         chunks.append({
                             "text": chunk,
                             "page_no": page_no
                         })

#                 for page_text in page_texts:
#                     chunks.extend(chunker.chunk(page_text))

                 print(f"Pages extracted: {len(page_texts)}")
                 print(f"Chunks created: {len(chunks)}")

                 for i, chunk in enumerate(chunks[:10]):
                    print(
                        f"Chunk {i+1} | page={chunk['page_no']} | "
                        f"text_length={len(chunk['text'])}"
                    )
                 print(f"Pages extracted: {len(page_texts)}")
                 print(f"Chunks created: {len(chunks)}")
                 for i, c in enumerate(chunks):
                    print(
                        f"Chunk {i+1} | page={c['page_no']} | "
                        f"text_length={len(c['text'])}"
                    )

                 # Debug logging for empty chunks
                 logger.info(f"DEBUG: Extracted {len(chunks)} chunks for {info['filename']}")
                 if chunks and not chunks[0]:
                     logger.error(f" Text extraction returned empty string for {info['filename']}")
                 print("=" * 100)
                 print("INGEST START")
                 print("CLIENT_ID:", client_id)
                 print("INDEX_NAME:", full_index_name)
                 print("CATEGORY_ID:", category_id)
                 print("FILE_NAME:", info["filename"])
                 print("FILE_UUID:", info["file_uuid"])
                 print("TOTAL_CHUNKS:", len(chunks))
                 if chunks:
                     print("FIRST_CHUNK_PAGE:", chunks[0]["page_no"])
                     print("FIRST_CHUNK_TEXT:")
                     print(chunks[0]["text"][:500])
                 print("=" * 100)
                 await store_pdf_chunks_in_redis(
                     index_name=full_index_name, 
                     pdf_file_name=info['filename'],
                     chunks=chunks,
                     redis_client=redis_client,
                     s3_location=s3_loc,
                     user_id=user_id,
                     category_id=category_id,
                     file_uuid=info['file_uuid']
                 )
                 try:
                     result = await redis_client.ft(full_index_name).search(
                         f'@category_id:"{category_id}"')
                     print("=" * 100)
                     print("POST INGEST CHECK")
                     print("INDEX:", full_index_name)
                     print("CATEGORY:", category_id)
                     print("DOC COUNT:", result.total)
                     print("=" * 100)
                 except Exception as e:
                     print("POST INGEST CHECK FAILED:", str(e))
                 db_record = QABOTParser(
                    client_id=client_id,
                    user_id=user_id,
                    file_uuid=info['file_uuid'],
                    file_name=info['filename'],
                    status="active",
                    s3_location=s3_loc,
                    created_by=user_id,
                    category_id=category_id,
                    file_type=get_file_type(info['filename']),
                    file_size=info['file_size'],
                    index_name=full_index_name,
                    pages=len(chunks)
                )
                 print("=" * 80)
                 print("DB RECORD")
                 print("category_id:", category_id, len(str(category_id)))
                 print("file_uuid:", info["file_uuid"], len(str(info["file_uuid"])))
                 print("status:", "active", len("active"))
                 print("file_type:", get_file_type(info["filename"]),
                      len(str(get_file_type(info["filename"]))))
                 print("=" * 80)
                 db.add(db_record)
                 db.commit()
                 processed_any = True
             except Exception as e:
                 db.rollback()          
                 logger.error(f"Error processing {info.get('filename')}: {e}")
                 continue
         if processed_any:
             return {"status": "success", "message": f"Indexed into {full_index_name}"}
         return {"status": "error", "message": "No files processed."}
    except Exception as e:
         db.rollback()
#         logger.error(f"Global Index Error: {e}")
         logger.exception("Global Index Error")
         return {"status": "error", "message": str(e)}


#@app.post("/query/")
#async def query_api(request: QueryRequest):
#    try:
        # request.user_query picks up 'question' alias automatically
#        user_query = request.user_query 
        
#        if not user_query:
#            return {"status": "error", "message": "No question provided. Use 'question' or 'user_query' field."}

#        target_language = detect_query_language(request.user_query)

 #       full_index_name = f"{request.client_id}_{request.index_name}"
 #       logger.info(
#            f"INDEX REQUEST | full_index_name={full_index_name} "
#            f"| client_id={client_id} "
#            f"| category_id={category_id} "
#            f"| embedding_dim={embedding_dim}"
#        )

#        logger.info(f"Querying: {user_query} on {full_index_name}")

#        indexes = await redis_client.execute_command("FT._LIST")
#        existing_indexes = [i.decode() if isinstance(i, bytes) else i for i in indexes]
        
#        if full_index_name not in existing_indexes:
#            return {
#                "status": "error", 
#                "message": f"Index {full_index_name} not found.",
#                "available_indexes": existing_indexes
#            }

#        redis_search = RedisSearch(redis_client, GOOGLE_API_KEY)
        #
#        history_key = f"chat_history:{request.session_id}"
#        chat_history = await redis_client.lrange(history_key, 0, -1)
#        chat_history_str = "\n".join([msg.decode() if isinstance(msg, bytes) else msg for msg in chat_history])
        
#        analysis = await redis_search.unified_query_analysis(chat_history_str, user_query)
#        reframed_q = analysis.get("reframed_question", user_query)
#        query_tag = analysis.get("query_tag", "Exact")

#        async def run_search(target_cid, target_answer_type):
#            if query_tag == "Exact":           
#                return await redis_search.query_search(
#                    reframed_q,
#                    request.user_id,
#                    request.category_id,
#                    target_cid,
#                    full_index_name,
#                    target_answer_type,
#                    target_language=target_language
#                )   

#            return await redis_search.research_query_search_with_subqueries(
#                reframed_q,
#                request.user_id,
#                request.category_id,
#                target_cid,
#                full_index_name,
#                target_answer_type,
#                analysis.get("sub_queries", []),
#                target_language=target_language   # âœ… ADD THIS
#            )

#        tasks = []
#        if request.client_id: tasks.append(run_search(request.client_id, request.answer_type or "INTERNAL"))
#        if request.master_client: tasks.append(run_search(request.master_client, "INTERNAL"))

#        results = await asyncio.gather(*tasks)
        
#        client_ans, client_docs = results[0] if request.client_id else ("", [])
#        print("=" * 80)
#        print("CLIENT DOCS RECEIVED IN MAIN")
#        print("Total Docs:", len(client_docs))

#        for i, doc in enumerate(client_docs):
#            print(
#                f"Doc {i+1} | page={doc.get('page_no')} | score={doc.get('score')}"
#            )

#        master_ans, master_docs = results[1] if len(results) > 1 else ("", [])
        # -------------------------------------------------------
        # Search only client documents
        # -------------------------------------------------------

#        client_ans = ""
#        client_docs = []
#        master_ans = ""
#        master_docs = []
#        if request.client_id:
#            client_ans, client_docs = await run_search(
#                request.client_id,
#                request.answer_type or "INTERNAL")
#        print("=" * 80)
#        print("CLIENT DOCS RECEIVED IN MAIN")
#        print("Total Docs:", len(client_docs))
#        for i, doc in enumerate(client_docs):
#            print(
#                f"Doc {i+1} | "
#                f"page={doc.get('page_no')} | "
#                f"score={doc.get('score')}")#

# master intentionally disabled
#        master_ans = ""
#        master_docs = []
#        seen = set()
#        merged_docs = []#

#        for doc in (client_docs or []) + (master_docs or []):
#            key = (doc["pdf_name"], doc["page_no"], doc["text"][:50])
#            if key not in seen:
#                seen.add(key)
#                merged_docs.append(doc)
                        
#        if not merged_docs:
#            return {
#                "status": "error", 
#                "message": "No data found in Redis.", 
#                "debug_info": {
#                    "index_queried": full_index_name,
#                    "reframed_question": reframed_q,
#                    "query_tag": query_tag
#                }
#            }

#        def is_similar(a, b):
#            return a.lower().strip()[:150] == b.lower().strip()[:150]

#        if client_ans and master_ans:
#            if is_similar(client_ans, master_ans):
#                combined_answer = client_ans   
#            else:
#                combined_answer = await redis_search.combine_answer_from_both_sources(
#                    reframed_q,
#                    client_ans,
#                    master_ans,
#                    target_language=target_language
#                )

#        elif client_ans:
#            combined_answer = client_ans
#        elif master_ans:
#            combined_answer = master_ans
#        else:
#            combined_answer = "No answer found."

        
#        combined_answer = clean_final_answer(combined_answer)

        # THEN store history (after successful processing)
#        await redis_client.rpush(history_key, user_query)
        
#        return {
#            "gen_ai_answer": combined_answer.strip(),
#            "reframed_question": reframed_q,
#            "top_docs": merged_docs
#        }
#    except Exception as e:
#        logger.error(f"Query Error: {e}")
#        traceback.print_exc()
#        return {"status": "failed", "message": str(e)}


#@app.get("/user-files/{user_id}", response_model=List[FileDetail])
#def get_user_files(user_id: int, db: Session = Depends(get_db)):
#    print("### NEW VERSION RUNNING ###")
#    #files = db.query(QABOTParser).filter(QABOTParser.UserId == user_id).all()
#    files = db.query(QABOTParser).filter(QABOTParser.user_id == user_id).all()
#    if not files:
#        raise HTTPException(status_code=404, detail="No files found")
#    print(files[0].__dict__)
#    return [
#    {
#            "FileUUID": f.file_uuid,
#            "FileName": f.file_name,
#            "Status": f.status,
#            "CategoryId": str(f.category_id),
#            "FileType": f.file_type,
#            "FileSize": f.file_size,
#            "IndexName": f.index_name,
#        }
#        for f in files
#    ]
#@app.get("/user-files/{user_id}", response_model=List[FileDetail])
#
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

@app.get("/user-files/{user_id}")
def get_user_files(user_id: int, db: Session = Depends(get_db)):
    # 1. Fetch from Database
    files = db.query(QABOTParser).filter(QABOTParser.user_id == user_id).all()

    if not files:
        return []

    # 2. Debug print to confirm data exists (as seen in your terminal)
    print(f"### MAPPING DATA FOR USER {user_id} ###")

    # 3. Manual Mapping
    # IMPORTANT: The keys here (e.g., "FileUUID") must match 
    # the FileDetail class variables EXACTLY.
    results = []
    for f in files:
        results.append({
            "FileUUID": str(f.file_uuid),
            "FileName": str(f.file_name),
            "Status": str(f.status),
            "CategoryId": str(f.category_id), 
            "FileType": str(f.file_type),
            "FileSize": int(f.file_size) if f.file_size else 0,
            "IndexName": str(f.index_name)
        })

    return results
@app.post("/generate-presigned-url", response_model=S3Output)
def generate_presigned_url_api(payload: S3Input):
    url = get_presigned_url(str(payload.s3_url), payload.expiration, payload.disposition)
    return {"presigned_url": url}

@app.delete("/delete-file/{index_name}/{file_uuid}")
async def delete_file(index_name: str, file_uuid: str, db: Session = Depends(get_db)):
    try:
        from redis.commands.search.query import Query as RedisQuery
        file_uuid_escaped = file_uuid.replace('-', "\\-")
        query = f'@file_uuid:{{{file_uuid_escaped}}}'
        results = await redis_client.ft(index_name).search(RedisQuery(query).paging(0, 3000))
        for doc in results.docs:
            await redis_client.delete(doc.id)
        #db_obj = db.query(QABOTParser).filter_by(FileUUID=file_uuid).first()
        db_obj = db.query(QABOTParser).filter_by(file_uuid=file_uuid).first()
        if db_obj:
            #db_obj.Status = "Deleted"
            db_obj.status = "deleted"
            db.commit()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- MULTILINGUAL ENDPOINTS ---
@app.post("/generate-question-paper/")
async def generate_question_paper_api(
    knowledge_files: List[UploadFile] = File(...),
    qp_format_file: UploadFile = File(...),
    target_language: str = Form('en')
):
    try:
        status, output_file_path = await generate_question_paper_multilingual(knowledge_files, qp_format_file, target_language)
        if status == 'done':
            return StreamingResponse(open(output_file_path, "rb"), media_type="application/pdf")
        raise Exception(output_file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/image-answer/")
# async def image_query_api(
#     file: UploadFile = File(...),
#     session_id: str = Form(...),
#     index_name: str = Form(...),
#     client_id: str = Form(...),
#     user_id: str = Form(None),
#     category_id: str = Form(None),
#     master_client: str = Form(None),
#     answer_type: str = Form(None),
#     target_language: str = Form('en')
# ):
#     try:
#         image_bytes = await file.read()
#         question_lists = await image_question_extraction_multilingual(image_bytes, target_language)
#         if not question_lists: return {"status": "failed"}

#         async def process_question(q):
#             req = QueryRequest(
#                 user_query=q, session_id=session_id, index_name=index_name, client_id=client_id, 
#                 user_id=user_id, category_id=category_id, master_client=master_client, answer_type=answer_type,
#                 target_language=target_language
#             )
#             result = await query_api(req)
#             return {"question": q, "answer": result.get("gen_ai_answer", "No answer")}

#         results = await asyncio.gather(*[process_question(q) for q in question_lists])
#         combined = "".join([f"<h3>{r['question']}</h3><p>{r['answer']}</p>" for r in results])
#         return {"status": "success", "gen_ai_answer": combined}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@app.post("/image-answer/")
async def image_query_api(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    index_name: str = Form(...),
    client_id: str = Form(...),
    user_id: str = Form(None),
    category_id: str = Form(None),
    master_client: str = Form(None),
    answer_type: str = Form(None),
##    target_language: str = Form('en')
):
    try:
        image_bytes = await file.read()

        # ðŸ” Extract questions from image
##        question_lists = await image_question_extraction_multilingual(image_bytes, target_language)
        question_lists = await image_question_extraction_multilingual(image_bytes)
        print("ðŸ”¥ Extracted Questions:", question_lists)

        if not question_lists:
            return {
                "status": "failed",
                "message": "No questions extracted from image"
            }

        # ðŸ”¥ Process each question safely
        async def process_question(q):
            try:
                if isinstance(q, dict):
                    question_text = (
                        q.get("query")
                        or q.get("query_original")
                        or q.get("original")
                        or q.get("question")
                        or q.get("reframed")
                        or q.get("query_en")
                        or q.get("translation")
                    )
                else:
                    question_text = str(q)

                if not question_text:
                    print("âš ï¸ Skipping empty question:", q)
                    return None

                question_text = question_text.strip()
                question_text = re.sub(r'^\s*\d+\s*[\.\)\-:]*\s*', '', question_text).strip()

                detected_lang = detect_query_language(question_text)

                print(f"Detected Lang for question: {detected_lang}")

                req = QueryRequest(
                    user_query=question_text,
                    session_id=session_id,
                    index_name=index_name,
                    client_id=client_id,
                    user_id=user_id,
                    category_id=category_id,
                    master_client=master_client,
                    answer_type=answer_type,
                    target_language=detected_lang
                )

#                result = await query_api_stream(req)

#                answer = result.get("gen_ai_answer", "No answer found")
##                answer = answer.strip()

#                client_urls = result.get("client_urls", [])
#                master_urls = result.get("master_urls", [])

##                all_urls = list(set(client_urls + master_urls))
#                all_urls = []
#                seen = set()

#                for url in client_urls + master_urls:
#                    if url not in seen:
#                        seen.add(url)
#                        all_urls.append(url)
                redis_search = RedisSearch(redis_client, GOOGLE_API_KEY)
                full_index_name = f"{req.client_id}_{req.index_name}"
                analysis = await redis_search.unified_query_analysis(
                    "",req.user_query)
                current_answer_type = analysis.get("answer_type", "other")
                reframed_question = analysis.get("reframed_question",req.user_query)
                query_tag = analysis.get("query_tag","Exact")
                sub_queries = analysis.get("sub_queries",[])
                async def run_search():
                    if query_tag == "Exact":
                        return await redis_search.query_search(
                            reframed_question,
                            req.user_id,
                            req.category_id,
                            req.client_id,
                            full_index_name,
                            current_answer_type,
                            target_language=detected_lang
                        )
                    else:
                        return await redis_search.research_query_search_with_subqueries(
                            reframed_question,
                            req.user_id,
                            req.category_id,req.client_id,full_index_name,current_answer_type,
                            sub_queries,
                            target_language=detected_lang
                        )
                client_answer = ""
                client_docs = []
                master_answer = ""
                master_docs = []
                client_answer, client_docs = await run_search()
                MAX_RETRY = 2
                result = {"gen_ai_answer": get_fallback_message(detected_lang),
                    "urls": []}
                for attempt in range(MAX_RETRY):
                    print(f"IMAGE QUERY Attempt {attempt+1}")
                    if not client_docs:
                        if attempt == 0:
                            print("Retrying due to empty docs...")
                            await asyncio.sleep(0.2)
                            client_answer, client_docs = await run_search()
                            continue
                        
                        result = {"gen_ai_answer": get_fallback_message(detected_lang),"urls": []}
                        break

                    combined_answer =await redis_search.combine_answer_from_both_sources(
                            reframed_question,client_answer,master_answer,
                            target_language=detected_lang)
                    print("=" * 80)
                    print("COMBINED ANSWER")
                    print(repr(combined_answer))
                    print("=" * 80)
                    if not combined_answer or not combined_answer.strip():
                        if attempt == 0:
                            continue
                        result ={"gen_ai_answer": get_fallback_message(detected_lang),"urls": []}
                        reak
                    best_rerank_score = (
                        client_docs[0].get("_rerank_score", 0)
                        if client_docs else 0)
                    if is_fallback_response(combined_answer):
                        if best_rerank_score >= 0.90:
                            pass
                        else:
                            result ={"gen_ai_answer": get_fallback_message(detected_lang), "urls": []}
                            break
                    client_urls = extract_urls_from_docs(client_docs)
                    result = {
                        "gen_ai_answer": combined_answer,
                        "client_urls": client_urls,"master_urls": []}

                    break
                answer = result.get("gen_ai_answer","No answer found")
#                all_urls = result.get( "urls",[])
                all_urls = (result.get("client_urls", []) + result.get("master_urls", []))
                if all_urls:
                    ref_html = "<div><b>References:</b>"
                    for i, url in enumerate(all_urls, 1):
                        ref_html += f'<br>[{i}] <a href="{url}">Reference {i}</a>'
                    ref_html += "</div>"

            # IMPORTANT: close paragraph before adding div
              #      answer += "</p>" + ref_html


                return {
                    "question": question_text,
                    "gen_ai_answer": answer,
##                    "gen_ai_answer": result.get("gen_ai_answer", "No answer Found")
                    "urls": all_urls
                }
            except Exception as e:
                print("Error processing question:", e)
                return None


        # Run all questions
        results = await asyncio.gather(*[process_question(q) for q in question_lists])

        print("Raw Results:", results)

        # Remove None / invalid results
        results = [r for r in results if r and r.get("question")]

        if not results:
            return {
                "status": "failed",
                "questions": "" ,
                "gen_ai_answer": "No valid answers generated"
            }

        combined_text = ""
        all_urls = set()
        for item in results:
            question = item.get("question", "")
            answer = item.get("gen_ai_answer", "No answer found")
#            all_urls = item.get("urls", [])
            answer = answer.replace("<p>", "").replace("</p>", "").strip()
            combined_text += f"<h1 style='font-size:24px'>{question}</h1>"
            lines = answer.split("\n")
            if lines:
                lines[0] = f"<b>{lines[0]}</b>"
            answer = "<br>".join(lines)
            combined_text += f"<div>{answer}</div><br>"
            urls = item.get("urls") or []
            all_urls.update(urls)


        if all_urls:
#            combined_text += "<p><b>References:</b></p>"
#            for i, url in enumerate(sorted(all_urls), 1):
#                combined_text += f'<p>[{i}] <a href="{url}">Reference {i}</a></p>'
            combined_text += "<br><b>References:</b> "
            for i, url in enumerate(sorted(all_urls), 1):
                combined_text += (
                    f'<a href="{url}" '
                    f'target="_blank" '
                    f'style="margin-right:8px;">[{i}]</a>'
                )
        return {
            "status": "success",
            "total_questions": [r["question"] for r in results],
            "gen_ai_answer": combined_text
        }

    except Exception as e:
        print("IMAGE API ERROR:", str(e))
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/ragas/health")
async def ragas_health():
    """
    Lightweight health check. It does not call Gemini and therefore does not
    consume model quota.
    """
    try:
        import ragas
        return {
            "status": "ok",
            "ragas_version": getattr(ragas, "__version__", "unknown"),
            "evaluator_model": os.getenv(
                "RAGAS_EVALUATOR_MODEL", "gemini-2.5-flash"
            ),
            "auto_evaluate": os.getenv(
                "RAGAS_AUTO_EVALUATE", "false"
            ).lower() == "true",
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ragas is not available: {exc}",
        )


@app.post("/ragas/evaluate-response")
async def ragas_evaluate_response(request: RagasResponseRequest):
    """
    Evaluate an existing answer.

    Use this when you already have:
      user question + answer + retrieved contexts.

    This endpoint does not execute Redis retrieval.
    """
    try:
        result = await evaluate_rag_response(
            google_api_key=GOOGLE_API_KEY,
            user_input=request.user_query,
            response=request.response,
            docs=[{"text": context} for context in request.retrieved_contexts],
            reference=request.reference_answer,
            evaluator_model=request.evaluator_model,
        )

        return {
            "status": "success",
            "evaluation": result,
        }

    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": str(exc),
            },
        )


@app.post("/ragas/evaluate-query")
async def ragas_evaluate_query(request: RagasEvaluateRequest):
    """
    End-to-end evaluation.

    It runs the SAME query analysis/retrieval/generation path as the normal
    RAG endpoint, then evaluates the resulting answer and retrieved chunks.

    This is the recommended local QA/regression endpoint.
    """
    try:
        detected_lang = normalize_lang(
            detect_query_language(request.user_query)
        )

        full_index_name = f"{request.client_id}_{request.index_name}"

        redis_search = RedisSearch(
            redis_client,
            GOOGLE_API_KEY,
        )

        # 1. Same canonicalization/classification used by production RAG.
        analysis = await redis_search.unified_query_analysis(
            "",
            request.user_query,
        )

        answer_type = (
            request.answer_type
            or analysis.get("answer_type")
            or "other"
        )
        reframed_question = analysis.get(
            "reframed_question",
            request.user_query,
        )
        query_tag = analysis.get("query_tag", "Exact")
        sub_queries = analysis.get("sub_queries", [])

        # 2. Same retrieval + generation functions used by the API.
        if query_tag == "Exact":
            answer, docs = await redis_search.query_search(
                reframed_question,
                request.user_id,
                request.category_id,
                request.client_id,
                full_index_name,
                answer_type,
                target_language=detected_lang,
            )
        else:
            answer, docs = await redis_search.research_query_search_with_subqueries(
                reframed_question,
                request.user_id,
                request.category_id,
                request.client_id,
                full_index_name,
                answer_type,
                sub_queries,
                target_language=detected_lang,
            )

        # 3. Evaluate the real output.
        evaluation = await evaluate_rag_response(
            google_api_key=GOOGLE_API_KEY,
            user_input=request.user_query,
            response=answer or "",
            docs=docs or [],
            reference=request.reference_answer,
            evaluator_model=request.evaluator_model,
        )

        return {
            "status": "success",
            "query": request.user_query,
            "reframed_question": reframed_question,
            "query_tag": query_tag,
            "answer_type": answer_type,
            "answer": answer,
            "retrieved_doc_count": len(docs or []),
            "evaluation": evaluation,
        }

    except Exception as exc:
        logger.exception("End-to-end RAGAS evaluation failed")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": str(exc),
            },
        )


@app.post("/query/stream/")
async def query_api_stream(request: QueryRequest, db: AsyncSession = Depends(get_db)):
    print("="*100)
    print("RAW REQUEST:", request.dict())
    print("="*100)

    logger.info("STEP 1: CACHE CHECK START")
    logger.info(
        f"QUERY REQUEST | "
        f"index_name={request.index_name} | "
        f"user_id={request.user_id} | "
        f"category_id={request.category_id} | "
        f"client_id={request.client_id}"
    )

    try:
        start_time = time.time()

        indexes = await redis_client.execute_command("FT._LIST")
        print(indexes)

    #    answer_type = request.answer_type if request.answer_type else "INTERNAL"
        answer_type = "other"
        user_id = request.user_id
        category_id = request.category_id
        full_index_name = f"{request.client_id}_{request.index_name}"
        user_query = request.user_query
        client_id = request.client_id
        master_client = request.master_client
        print("=" * 80)
        print("REQUEST SUBJECT:", getattr(request, "subject_id", None))
        print("REQUEST CATEGORY:", request.category_id)
        print("REQUEST INDEX:", request.index_name)
        print("REQUEST QUERY:", request.user_query)
        print("RAW REQUEST:", request.dict())
        print("=" * 80)
        # -----------------------------
        # Chat history
        # -----------------------------
        history_key = f"chat_history:{request.session_id}"
        chat_history = await redis_client.lrange(history_key, 0, -1)

        chat_history_str = "\n".join([
            msg.decode() if isinstance(msg, bytes) else msg
            for msg in chat_history
        ])
        redis_search = RedisSearch(redis_client, GOOGLE_API_KEY)

        raw_lang = detect_query_language(user_query)
        detected_lang = normalize_lang(raw_lang)

        print("Detected Language (raw):", raw_lang)
        print("Detected Language (final):", detected_lang)
        # -----------------------------
        # Unified analysis
        # -----------------------------
        analysis = await redis_search.unified_query_analysis(
            chat_history_str, user_query
        )
        answer_type = analysis.get("answer_type", "other")
        print(analysis)

        reframed_question = analysis.get("reframed_question", user_query)
        query_tag = analysis.get("query_tag", "Exact")
        sub_queries = analysis.get("sub_queries", [])

        print(f"Reframed: {reframed_question}, Tag: {query_tag}, Sub-queries: {sub_queries}")
        logger.info(
            f"CLASSIFIER OUTPUT | "
            f"query_tag={query_tag} | "
            f"answer_type={answer_type} | "
            f"reframed_question={reframed_question}"
        )

        # -----------------------------
        # Search Function
        # -----------------------------
        async def run_search(client_id, answer_type, target_language="en"):
            if query_tag == "Exact":
                return await redis_search.query_search(
                    reframed_question,
                    user_id,
                    category_id,
                    client_id,
                    full_index_name,
                    answer_type,
                    target_language=detected_lang  # internal only
                )
            else:
                return await redis_search.research_query_search_with_subqueries(
                    reframed_question,
                    user_id,
                    category_id,
                    client_id,
                    full_index_name,
                    answer_type,
                    sub_queries,
                    target_language=detected_lang
                )

        logger.info("STEP 2: RETRIEVAL START")
#        client_task = run_search(client_id, answer_type, target_language=detected_lang) if client_id else None
#        master_task = run_search(master_client, answer_type, target_language=detected_lang) if master_client else None

        # 2. Gather the results
#        results_list = await asyncio.gather(
#            *(t for t in [client_task, master_task] if t is not None))

         # 3. Carefully unpack the results based on which tasks were actually run
#        client_answer, client_docs = ("", [])
#        master_answer, master_docs = ("", [])

#        if client_id and master_client:
#            client_answer, client_docs = results_list[0]
#            master_answer, master_docs = results_list[1]
#        elif client_id:
#            client_answer, client_docs = results_list[0]
#        elif master_client:
#            master_answer, master_docs = results_list[0]
        client_answer = ""
        client_docs = []
        master_answer = ""
        master_docs = []
        if client_id:
            client_answer, client_docs = await run_search(
                client_id,
                answer_type,
                target_language=detected_lang)
        print("=" * 80)
        print("STREAM API DOCS")

        logger.info(
            f"GENERATION INPUT | "
            f"client_docs={len(client_docs)} | "
            f"master_docs={len(master_docs)} | "
            f"query_tag={query_tag} | "
            f"answer_type={answer_type}"
        )

        print("CLIENT DOCS:", len(client_docs))
        print("MASTER DOCS:", len(master_docs))

        for i, doc in enumerate(client_docs[:5]):
            print(
                f"Client Doc {i+1} | score={doc.get('score')} | page={doc.get('page_no')}"
            )

#        for i, doc in enumerate(master_docs[:5]):
#            print(f"Master Doc {i+1} | score={doc.get('score')} | page={doc.get('page_no')}")
# 4. Extract URLs
#        client_urls = extract_urls_from_docs(client_docs)
#        master_urls = extract_urls_from_docs(master_docs)
#        all_urls = list(set(client_urls + master_urls))
        all_urls = extract_urls_from_docs(client_docs)
        print("CLIENT ANSWER:\n", client_answer)
#        print("MASTER ANSWER:\n", master_answer)

    #    print("CLIENT DOCS:\n", client_docs)
    #    print("MASTER DOCS:\n", master_docs)
        # -----------------------------
        # Streaming Response
        # -----------------------------
        async def answer_stream():
            try:
                already_sent = False
                # -----------------------------
                #  CACHE KEY
                # -----------------------------
                analysis = await redis_search.unified_query_analysis(
                    "",
                    user_query
                )
                answer_type = analysis.get(
                    "answer_type",
                    "other"
                )
                canonical_query = analysis["reframed_question"]

                cache_key = (
                    f"qa:"
                    f"{request.client_id}:"
                    f"{full_index_name}:"
                    f"{analysis['query_tag']}:"
                    f"{normalize_query(canonical_query)}"
                )
                logger.info(
                    f"CACHE INPUT | "
                    f"query_tag={analysis['query_tag']} | "
                    f"answer_type={answer_type} | "
                    f"canonical_query={canonical_query}"
                )

                print("=" * 80)
                print("CACHE LOOKUP START")
                print("USER QUERY:", repr(user_query))
                print("CANONICAL QUERY:", repr(canonical_query))
                print("CACHE KEY:", cache_key)

                print(
                    "EXISTS:",
                    await redis_client.exists(cache_key)
                )
                cached = await redis_client.get(cache_key)

                print("CACHE FOUND:", cached is not None)

                if cached:
                    print("EXACT CACHE HIT")
                    yield cached.decode()
                    return

                print("EXACT CACHE MISS")
                print("=" * 80)
                semantic_answer = await semantic_cache_search(
                    redis_client,
                    query=canonical_query,
                    answer_type=answer_type,
                    threshold=0.91
                )

                if semantic_answer:
                    print("SEMANTIC CACHE HIT")
                    await redis_client.set(
                        cache_key,
                        semantic_answer,
                        ex=3600
                    )
                    yield semantic_answer
                    return
                print("CACHE MISS")

        # -----------------------------
        #  RETRY LOGIC
        # -----------------------------
                MAX_RETRY = 2

                for attempt in range(MAX_RETRY):

                    print(f" Attempt {attempt + 1}")
                    if already_sent:   # âœ… PREVENT DUPLICATE
                        return
                # -----------------------------
                # 1. Early fallback (NO DOCS)
                # -----------------------------
#                    if not client_docs and not master_docs:
                    if not client_docs:
                        if attempt == 0:
                            print("Retrying due to empty docs...")
                            await asyncio.sleep(0.2)
                            continue
                        else:
                            fallback_text = get_fallback_message(detected_lang)
                            final_html = f"<p>{fallback_text}</p>"
                            yield final_html
                            return
                # -----------------------------
              # âœ… SMART SHORT-CIRCUIT (AVOID LLM)
                # -----------------------------
#                    if client_answer and not master_answer:
#                        print("DEBUG: Using client_answer directly (skip combine)")
#                        combined_answer = client_answer

#                    elif master_answer and not client_answer:
#                        print("DEBUG: Using master_answer directly (skip combine)")
#                        combined_answer = master_answer

#                    else:
                # Only when BOTH exist â†’ use LLM
                    combined_answer = await redis_search.combine_answer_from_both_sources(
                        reframed_question,
                        client_answer,
                        master_answer,
                        target_language=detected_lang
                    )
                    print("=" * 80)
                    print("COMBINED ANSWER")
                    print(repr(combined_answer))
                    print("IS FALLBACK:", is_fallback_response(combined_answer))
                    print("=" * 80)    

                    logger.info(
                        f"ANSWER PATH | "
                        f"client_answer={'Y' if client_answer else 'N'} | "
                        f"master_answer={'Y' if master_answer else 'N'} | "
                        f"client_docs={len(client_docs)} | "
                        f"master_docs={len(master_docs)}"
                    )

                    if client_docs and not master_docs:
                        print("DEBUG: Docs exist but master empty â†’ forcing answer")

                    if not combined_answer or not combined_answer.strip():
                        if attempt == 0:
                            print("Retry due to empty answer...")
                            continue
                        else:
                            fallback_text = get_fallback_message(detected_lang)
                            final_html = f"<p>{fallback_text}</p>"
                            yield final_html
                            return
                # -----------------------------
        # ðŸš¨ CRITICAL FIX
        # If LLM says fallback but docs exist â†’ IGNORE fallback
        # -----------------------------
                    print("=" * 80)
                    print("FINAL ANSWER BEFORE FALLBACK CHECK")
                    print(combined_answer)
                    print("=" * 80)

                    best_rerank_score = client_docs[0].get("_rerank_score", 0) if client_docs else 0

                    if is_fallback_response(combined_answer):
                    #     if client_docs or master_docs:
                    #         print("BLOCKED FALLBACK: Docs exist, forcing answer")

                    #         if client_answer and client_answer.strip():
                    #             combined_answer = client_answer

                    #         elif master_answer and master_answer.strip():
                    #             combined_answer = master_answer

                    #         else:
                    #     # last safety fallback from docs
                    #             combined_answer = client_docs[0]["text"][:500]

                    #     else:
                    #         fallback_text = get_fallback_message(detected_lang)
                    #         final_html = f"<p>{fallback_text}</p>"
                    #         yield final_html
                    #         return

                    # if is_fallback_response(combined_answer):

                    #     docs_exist = (
                    #         (client_docs and len(client_docs) > 0)
                    #         or
                    #         (master_docs and len(master_docs) > 0)
                    #     )

                    #     if docs_exist:

                    #         best_doc = None

                    #         if client_docs:
                    #             best_doc = client_docs[0]

                    #         elif master_docs:
                    #             best_doc = master_docs[0]

                    #         if best_doc:
                    #             combined_answer = best_doc["text"][:600]
                        if best_rerank_score >= 0.90:
                            # allow answer
                            pass
                        else:
                            fallback_text = get_fallback_message(detected_lang)
                            final_html = f"<p>{fallback_text}</p>"
                            yield final_html
                            return

                reference_links = []
                if not is_fallback_response(combined_answer):
                    all_docs = client_docs
                    seen = set()

                    for doc in all_docs:
                        pdf_name = doc.get("pdf_name", "")
                        page_no = str(doc.get("page_no", "")).strip()
                        unique_key = f"{pdf_name}_{page_no}"
                        if unique_key in seen:
                            continue
                        seen.add(unique_key)
                        url = doc.get("url", "").strip()
                        if not url:
                            continue
                        if page_no and page_no.isdigit():
                            full_url = f"{url}#page={page_no}"
                        else:
                            full_url = url
                        reference_links.append(full_url)
                        print("REFERENCE:",pdf_name,"PAGE:",page_no,"URL:",full_url)

                formatted_answer = combined_answer.strip()
                    # Remove markdown ##
                formatted_answer = formatted_answer.replace("##", "")
                    # Preserve structure
                formatted_answer = formatted_answer.replace("\n", "<br>")

        # -----------------------------
        # Create References HTML
        # -----------------------------
                ref_html = ""

                if(not is_fallback_response(combined_answer) and reference_links):
                    ref_html = "<div><b>References:</b> "

                    for i, link in enumerate(reference_links, 1):
                        ref_html += (
                            f'<a href="{link}" '
                            f'target="_blank" '
                            f'style="margin-right:8px;">[{i}]</a>'
                        )

                final_html = f"<p>{formatted_answer}</p>{ref_html}"
                print("IS FALLBACK:", is_fallback_response(combined_answer))
                    # if not is_fallback_response(combined_answer):
                    #     print("ENTERING CACHE STORE BLOCK")
                    #     await redis_client.set(cache_key, final_html, ex=3600)
                    #     print("CALLING STORE SEMANTIC CACHE") 
                    #     if not is_fallback_response(combined_answer):  
                    #         await store_semantic_cache(
                    #             redis_client,
                    #             cache_key=cache_key,
                    #             query=canonical_query,
                    #             answer_type=answer_type
                    #         )             
                    #     print("STORE SEMANTIC CACHE COMPLETED")

#                docs_exist = (len(client_docs) > 0 or len(master_docs) > 0)
                docs_exist = len(client_docs) > 0
                fallback = is_fallback_response(combined_answer)

                if not docs_exist:
                    print("SKIP CACHE: NO DOCS")

                elif fallback:
                    print("SKIP CACHE: FALLBACK ANSWER")

                    #elif not is_fallback_response(combined_answer):
                else:
                    print("ENTERING CACHE STORE BLOCK")

                    await redis_client.set(
                        cache_key,
                        final_html,
                        ex=3600
                    )

                    print("CALLING STORE SEMANTIC CACHE")

                    await store_semantic_cache(
                        redis_client,
                        cache_key,
                        canonical_query,
                        answer_type
                    )

                    print("STORE SEMANTIC CACHE COMPLETED")

                if not already_sent:
                    already_sent = True
                    yield final_html
                return

            except Exception as e:
                print("STREAM ERROR:", str(e))
                yield f"data: Error: {str(e)}\n\n"

        # -----------------------------
        # Merge docs (optional)
        # -----------------------------
#        merged_docs = []
#        if client_docs:
#            merged_docs.extend(client_docs)
#        if master_docs:
#            merged_docs.extend(master_docs)
        merged_docs = client_docs.copy()
        # -----------------------------
        # Save history
        # -----------------------------
        await redis_client.rpush(history_key, user_query)

        end_time = time.time()
        print("Execution time:", end_time - start_time, "seconds")

        return StreamingResponse(
            answer_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
                "Content-Encoding": "none"
            }
        )

    except Exception as e:
        print(traceback.print_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "status": "failed",
                "message": f"Something went wrong: {str(e)}"
            }
        )

@app.post("/summarise-file/")
async def summarise_file_api(file: UploadFile = File(...), target_language: str = Form('en')):
    try:
        tmp_dir = "tmp_index"
        file_infos = read_and_prepare_file(file, tmp_dir)
        all_texts = []
        for info in file_infos:
            processor = PDFProcessor(info['local_file_path'], info['file_uuid'])
            text = processor.process_pdf() if info['ext'] == ".pdf" else processor.process_image()
            all_texts.append(str(text))
        summary = await get_llm_summary_multilingual("\n".join(all_texts), target_language)
        return {"status": "success", "summary_html": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

