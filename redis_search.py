
import asyncio
import json
import re
import unicodedata

from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI
from redis.commands.search.query import Query as RedisQuery
from sentence_transformers import SentenceTransformer

from advanced_services.multilingual_pipeline import MultilingualPipeline
from advanced_services.query_decomposition import QueryDecomposer
from advanced_services.reranker import Reranker
from language_utils import (
    detect_query_language,
    get_fallback_message as localized_fallback_message,
    language_name,
    normalize_language_code,
)
from prompts import COMBINE_ANSWER_PROMPT
from s3 import get_presigned_url

embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base")
embedding_dim = 768
GEMINI_MODEL = "gemini-2.5-flash"

MIN_CONTEXT_SIMILARITY = 0.70
GOOD_CONTEXT_SIMILARITY = 0.75
KNN_CANDIDATES = 20
FINAL_DOC_LIMIT = 4


def detect_lang(text):
    return detect_query_language(text)


def _normalize_text(value: str) -> str:
    if not value:
        return ""

    value = value.lower()
    value = unicodedata.normalize("NFKD", value)
    value = re.sub(r"[^\w\s\u0900-\u097F]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _token_overlap(query: str, text: str) -> float:
    query_tokens = {t for t in _normalize_text(query).split() if len(t) > 2}
    text_tokens = {t for t in _normalize_text(text).split() if len(t) > 2}
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def _similarity_from_distance(distance) -> float:
    try:
        return max(0.0, min(1.0, 1.0 - float(distance)))
    except Exception:
        return 0.0


def _safe_json_from_text(text: str) -> dict:
    if not text:
        return {}
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    json_str = match.group(1) if match else text
    return json.loads(json_str)


def get_fallback_message_for_lang(lang):
    return localized_fallback_message(lang)


# Kept for backward compatibility with imports in main.py
get_fallback_message = get_fallback_message_for_lang


class RedisSearch:
    def __init__(self, redis_client, google_api_key):
        self.redis_client = redis_client
        self.GOOGLE_API_KEY = google_api_key
        self.multilingual_pipeline = MultilingualPipeline()
        self.query_decomposer = QueryDecomposer()
        self.reranker = Reranker()

    async def unified_query_analysis(self, chat_history, user_query):
        print("CALLING GEMINI: unified_query_analysis")

        try:
            prompt = f"""
You are an advanced multilingual RAG assistant. Given chat history and a user query:

1. Rewrite the query as a standalone canonical question in the same language.

2. Normalize synonyms, titles, and entity references.

3. Questions with the same meaning must produce the same reframed_question.
    You are a query canonicalization engine.

    For semantically equivalent questions,
    ALWAYS produce exactly the same reframed_question.

    Rules:

    - Use one canonical entity name.
    - Use one canonical wording.
    - Do not vary titles.
    - Do not vary honorifics.
    - Do not vary sentence structure.

    Examples:

    Input:
    शिवराय कधी जन्मले ?
    Output:
    शिवाजी महाराजांचा जन्म कधी झाला?

    Input:
    शिवरायांचा जन्म कधी झाला ?
    Output:
    शिवाजी महाराजांचा जन्म कधी झाला?

    Input:
    छत्रपती शिवाजी महाराजांचा जन्म कधी झाला ?
    Output:
    शिवाजी महाराजांचा जन्म कधी झाला?


4. Classify it as "Exact" or "Research".
    Query Classification Rules:

        Use "Exact" when the answer can reasonably be answered with a single fact,
        single explanation, or a small set of facts.

        Use "Research" when answering requires:
        - multiple aspects
        - detailed explanation
        - historical discussion
        - comparison
        - analysis
        - long-form informational content
        - multiple supporting points

        Examples:

        "शिवाजी महाराजांचा जन्म कधी झाला?"
        => Exact

        "शिवाजी महाराजांचा जन्म कुठे झाला?"
        => Exact

        "संत नामदेवांविषयी माहिती द्या"
        => Research

        "महात्मा गांधींबद्दल माहिती सांगा"
        => Research

        "भारताच्या स्वातंत्र्य चळवळीविषयी माहिती द्या"
        => Research

        "लोकशाही म्हणजे काय?"
        => Exact


        Use "Peoblem_solving" when :
        The user is asking to solve, derive, calculate,
        determine, infer, complete, evaluate, transform,
        analyze, classify, translate, generate, or apply
        information using knowledge, rules, procedures,
        examples, formulas, methods, or instructions
        contained in the retrieved documents.
        Examples:
        - Exercises
        - Practice questions
        - Numerical problems
        - MCQs
        - Fill in the blanks
        - Derivations
        - Conversions
        - Grammar corrections
        - Programming tasks
        - Reasoning questions
        - Case-study based questions

        Return:
        query_tag = Exact
        answer_type = problem_solving

5. Determine the expected answer type.

    Possible answer types:

    date
    location
    person
    number
    definition
    list
    yes_no
    reason
    process
    procedure
    comparsion
    problem_solving
    summary
    other

    Answer Type Rules:

        1. Use "date" when the user asks:

        * when something happened
        * birth date
        * death date
        * year, month, or day

        2. Use "location" when the user asks:

        * where
        * birthplace
        * place of an event
        * location of a person, place, or thing

        3. Use "person" when the answer is expected to be the name of a person.

        4. Use "number" when the answer is expected to be:

        * count
        * quantity
        * percentage
        * distance
        * area
        * population
        * measurement
        * numerical value
        * What is the atomic number of oxygen?
        * How many states are there in India?

        5. Use "definition" only when the user asks:

        * what is X
        * meaning of X
        * definition of X
        * brief explanation of a term or concept

        6. Use "list" only when the primary answer is expected to contain multiple items, such as:

        * types
        * names
        * examples
        * duties
        * rivers
        * states
        * causes
        * advantages
        * disadvantages
        * categories

        7. Use "yes_no" when the expected answer is yes or no.

        8. Use "reason" when the user asks why something happens.

        9. Use "process" when the user asks:

        * how something works
        * steps of a process
        * procedure

        10.Use problem_solving when the query requires:
         - calculation
         - derivation
         - reasoning
         - inference
         - applying formulas
         - applying methods
         - solving exercises
         - answering textbook questions
         - completing examples
         - working through procedures found in retrieved documents

         Examples:
         "Find HCF of 96 and 404"
         "Find LCM of 12 and 18"
         "Prime factorise 140"
         "Solve x² − 5x + 6 = 0"
         "Find acceleration"
         "Balance the equation"
         "Answer exercise 3"


        11.Use "other" for:

        * biography requests
        * information about a person
        * information about a place
        * information about an event
        * historical descriptions
        * detailed explanations
        * research-style questions
        * open-ended informational queries
        * "tell me about" queries
        * "information about" queries
    
        Examples:

        "लोकशाही म्हणजे काय?"
        => definition

        "महाराष्ट्रातील प्रमुख नद्या कोणत्या?"
        => list

        "पाऊस का पडतो?"
        => reason

        "मतदान प्रक्रिया कशी असते?"
        => process

        "शिवाजी महाराजांचा जन्म कुठे झाला?"
        => location

        "शिवाजी महाराजांचा जन्म कधी झाला?"
        => date

        "संत नामदेवांविषयी माहिती द्या"
        => other

        "महात्मा गांधींबद्दल माहिती सांगा"
        => other

        "छत्रपती शिवाजी महाराजांचे कार्य स्पष्ट करा"
        => other

        "भारताच्या स्वातंत्र्य चळवळीविषयी माहिती द्या"
        => other

        11.Biography questions should be classified as:

        query_tag = "Exact"
        answer_type = "other"

        Examples:

        "संत तुकाराम कोण होते?"
        => Exact, other

        "महात्मा गांधी कोण होते?"
        => Exact, other

        "संत नामदेव कोण होते?"
        => Exact, other    

6. For Research, create 3 to 5 focused sub-queries.

Return strict JSON only:

{{
  "reframed_question": "...",
  "query_tag": "Exact",
  "answer_type": "date",
  "sub_queries": []
}}


Chat History:
{chat_history}

User Query:
{user_query}
"""            
            
            client = genai.Client(api_key=self.GOOGLE_API_KEY)
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            result = _safe_json_from_text(response.text)
            query_tag = result.get("query_tag", "Exact")
            answer_type = result.get(
                "answer_type",
                "other"
            )
            
            if query_tag == "Research":
                answer_type = "research"
            elif answer_type == "list":
                query_tag = "Research"
                answer_type = "research"

            if query_tag not in {"Exact", "Research"}:
                query_tag = "Exact"

            reframed_question = result.get("reframed_question") or user_query
            original_lang = detect_query_language(user_query)
            reframed_lang = detect_query_language(reframed_question, default=original_lang)
            if reframed_lang != original_lang:
                reframed_question = user_query

            sub_queries = result.get("sub_queries") or []
            guarded_sub_queries = []
            for sub_query in sub_queries:
                if detect_query_language(sub_query, default=original_lang) == original_lang:
                    guarded_sub_queries.append(sub_query)
            if sub_queries and not guarded_sub_queries:
                guarded_sub_queries = self.query_decomposer.generate_sub_queries(user_query)

            print("USER QUERY:", user_query)
            print("REFRAMED:", reframed_question)

            if len(user_query.split()) <= 8:
            
                return {
                    "reframed_question": reframed_question,
                    "query_tag": query_tag,
                    "answer_type": answer_type,
                    "sub_queries": guarded_sub_queries,
                }

            return {
                "reframed_question": reframed_question,
                "query_tag": query_tag,
                "answer_type": answer_type,
                "sub_queries": guarded_sub_queries,
            }
        except Exception as e:
            print(f"Error in unified_query_analysis: {e}")
            return {
                "reframed_question": user_query,
                "query_tag": "Exact",
                "answer_type": "other",
                "sub_queries": [],
            }

    async def translate_to_english(self, text):
        try:
            if detect_query_language(text) == "en":
                return text

            prompt = f"Translate the following text to English. Return only the translated text:\n{text}"
            client = genai.Client(api_key=self.GOOGLE_API_KEY)
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            translated = (response.text or "").strip()
            if translated and len(translated.split()) <= max(30, len(text.split()) * 3):
                return translated
        except Exception as e:
            print("Translation to English failed:", e)
        return text

    async def translate_to_target(self, text, target_language):
        print("CALLING GEMINI: translate_to_target")
        target_language = normalize_language_code(target_language)
        if not text or target_language == "en":
            return text

        try:
            prompt = f"""
Translate the following text into {language_name(target_language)}.
Return only the translated text.
Do not add explanations, prefixes, URLs, or citations.

TEXT:
{text}
"""
            client = genai.Client(api_key=self.GOOGLE_API_KEY)
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )
            translated = (response.text or "").strip()
            return translated or text
        except Exception as e:
            print("Translation to target failed:", e)
            return text

    async def classify_query(self, query):
        return (await self.unified_query_analysis("", query)).get("query_tag", "Exact")

    async def get_sub_queries(self, query):
        sub_queries = (await self.unified_query_analysis("", query)).get("sub_queries", [])
        return sub_queries or self.query_decomposer.generate_sub_queries(query)

    async def _query_variants(self, query: str) -> list[str]:
        normalized = self.multilingual_pipeline.normalize(query)
        variants = [normalized["original_query"]]
        detected_lang = normalized["language"]
        if detected_lang != "en":
            translated = await self.translate_to_english(query)
            if translated and _normalize_text(translated) != _normalize_text(query):
                variants.append(translated)
        return list(dict.fromkeys(v for v in variants if v and v.strip()))

    async def _search_variant(self, query, user_id, category_id, index_name):
        loop = asyncio.get_running_loop()
        query_embedding = await loop.run_in_executor(
            None,
        #     lambda: embedding_model.encode(query).astype("float32").tobytes(),
        # )
                lambda: embedding_model.encode(f"query: {query}",normalize_embeddings=True).astype("float32").tobytes(),
        )
        filters = []
        #if user_id:
         #   filters.append(f'@user_id:"{user_id}"')
        if category_id:
            filters.append(f'@category_id:"{category_id}"')

        filter_str = " ".join(filters)
        if filter_str:
            redis_query = RedisQuery(f"({filter_str})=>[KNN {KNN_CANDIDATES} @embedding $vec AS score]")
        else:
            redis_query = RedisQuery(f"*=>[KNN {KNN_CANDIDATES} @embedding $vec AS score]")

        redis_query.sort_by("score").return_fields(
            "text",
            "pdf_name",
            "score",
            "url",
            "chunk_id",
            "page_no"
        ).dialect(2)

        print("=" * 80)
        print("SEARCH DEBUG")
        print("INDEX:", index_name)
        print("QUERY:", query)
       # print("USER_ID:", user_id)
        print("CATEGORY_ID:", category_id)
        print("FILTER:", filter_str)
        print("=" * 80)

        return await self.redis_client.ft(index_name).search(
            redis_query,
            query_params={"vec": query_embedding},
        )
        print("REDIS DOC COUNT:", len(result.docs))

        return result

    async def _retrieve_docs(self, query, user_id, category_id, index_name, answer_type="other"):
        variants = await self._query_variants(query)
        
        print("=" * 80)
        print("INSIDE _retrieve_docs")
        print("QUERY:", query)
        print("ANSWER TYPE RECEIVED:", answer_type)
        print("=" * 80)

        results = await asyncio.gather(
            *[self._search_variant(v, user_id, category_id, index_name) for v in variants],
            return_exceptions=True,
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print("SEARCH ERROR:", result)
            else:
                print(f"VARIANT {i} DOC COUNT:", len(result.docs))

        raw_docs = []
        for result in results:
            if isinstance(result, Exception):
                print("Redis variant search failed:", result)
                continue
            for doc in result.docs:
                print("=" * 80)
                print("AVAILABLE FIELDS:", doc.__dict__)
                print("=" * 80)
                pdf_name = doc.pdf_name.decode() if isinstance(doc.pdf_name, bytes) else doc.pdf_name
                text = doc.text.decode() if isinstance(doc.text, bytes) else doc.text
                url = doc.url.decode() if isinstance(doc.url, bytes) else doc.url
                chunk_id = doc.chunk_id.decode() if isinstance(doc.chunk_id, bytes) else doc.chunk_id
                distance = float(doc.score)
       #         page_no = chunk_id.split("_")[0] if "_" in chunk_id else chunk_id
                page_no = (doc.page_no.decode()if isinstance(doc.page_no, bytes)else doc.page_no)
                overlap = _token_overlap(query, f"{pdf_name} {text}")
                similarity = _similarity_from_distance(distance)
                rerank_score = similarity + min(0.12, overlap * 0.12)

                raw_docs.append(
                    {
                        "pdf_name": pdf_name,
                        "text": text,
                        "url": get_presigned_url(url),
                        "page_no": page_no,
                        "score": distance,
                        "_similarity": similarity,
                        "_rerank_score": rerank_score,
                    }
                )

        if not raw_docs:
            return []

        deduped = []
        seen = set()
        for doc in sorted(raw_docs, key=lambda d: d["_rerank_score"], reverse=True):
            key = (doc["pdf_name"], doc["page_no"], _normalize_text(doc["text"])[:160])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(doc)

        print("=" * 80)
        print("ALL RETRIEVED DOCS BEFORE FILTER")

        MAX_SIM = max(
            (d["_similarity"] for d in deduped),
            default=0
        )

        print("BEST SIMILARITY:", MAX_SIM)

        if answer_type in ["date", "location", "person", "number"]:
            min_similarity = 0.72

        elif answer_type in ["definition", "yes_no"]:
            min_similarity = 0.75

        else:
            min_similarity = 0.78

        print("MAX_SIM:", MAX_SIM)
        print("MIN_SIM:", min_similarity)
        print("ANSWER TYPE:", answer_type)
        for i, doc in enumerate(deduped[:10]):
            print(
                f"DOC {i}",
                "SIM=", doc.get("_similarity"),
                "PDF=", doc.get("pdf_name"),
                "PAGE=", doc.get("page_no")
            )

        if MAX_SIM < min_similarity:    
            print("NO RELEVANT DOCS FOUND")
            return []

        for d in deduped:
            print(
                f"similarity={d['_similarity']:.3f} "
                f"score={d['score']:.3f} "
                f"page={d['page_no']}"
            )

        strong_docs = [d for d in deduped if d["_similarity"] >= GOOD_CONTEXT_SIMILARITY]
        usable_docs = strong_docs or [
            d for d in deduped if d["_similarity"] >= MIN_CONTEXT_SIMILARITY or _token_overlap(query, d["text"]) >= 0.18
        ]

        if not usable_docs:
            print("No usable docs after context-quality filtering")
            return []

        final_docs = self.reranker.rerank(query, usable_docs)[:FINAL_DOC_LIMIT]
        print("\nTOP 10 AFTER VECTOR BEFORE RERANK")
        print("=" * 80)
        print("FINAL DOCS AFTER RERANK")
        for doc in final_docs:
            print(
                f"PAGE={doc.get('page_no')} "
                f"SIM={doc.get('_similarity', 0):.4f} "
                f"RERANK={doc.get('_rerank_score', 0):.4f}"
            )
            print(doc.get("text", "")[:400])
            print("-" * 80)

        for d in usable_docs[:10]:
            print(
                d.get("page_no"),
                d.get("_similarity"),
                d.get("text","")[:150]
            )

        if final_docs:

            best_score = final_docs[0].get(
                "_rerank_score",
                0
            )
            print(f"BEST RERANK SCORE={best_score:.4f}")
            
            if answer_type in ["date", "location", "person", "number"]:
                rerank_threshold = 0.70
            else:
                rerank_threshold = 0.80

            best_similarity = final_docs[0].get("_similarity", 0)
#            if best_score < rerank_threshold:
            if (best_score < rerank_threshold and best_similarity < min_similarity):
                print(
                    f"BEST DOC TOO WEAK: {best_score}"
                )
                return []
        print("=" * 80)
        print("RETRIEVED DOCS")
        print(f"Query: {query}")
        
        print("\nAFTER RERANK RETURN")
        for doc in final_docs:
            print(
                f"rerank={doc.get('_rerank_score')} "
                f"vector={doc.get('_vector_score')} "
                f"page={doc.get('page_no')}"
            )


        for i, doc in enumerate(final_docs):
            # print(
            #     f"Doc {i+1} | score={doc['score']:.4f} | chars={len(doc['text'])}"
            # )
            print(
                f"Doc {i+1} | rerank={doc.get('_rerank_score',0):.4f} "
                f"| vector={doc['score']:.4f}"
            )
            print(doc["text"][:200])
            print("-" * 40)

        for doc in final_docs:
            doc.pop("_similarity", None)
           # doc.pop("_rerank_score", None)
        print("\nTOP DOC TEXTS AFTER RERANK\n")
        for i, doc in enumerate(final_docs):
            print(f"\nDOC {i+1}")
            print(doc["text"][:500])
        return final_docs

    async def get_genai_answer(self, query, docs, answer_type, target_language="en"):
        print("CALLING GEMINI: get_genai_answer")
        target_language = normalize_language_code(target_language or detect_query_language(query))

        if not docs:
            return get_fallback_message(target_language)

        context = ""
        for idx, doc in enumerate(docs, start=1):
            context += (
                f"\n\n[Document {idx}]\n"
                f"{doc.get('text', '')}\n"
                f"Source URL: {doc.get('url', '')}#page={doc.get('page_no', '')}"
            )

        prompt = f"""
You are a strict retrieval-augmented assistant.

You are a document-grounded assistant.

Use the provided documents as the PRIMARY source of truth.

Do not use knowledge that is unrelated to the
retrieved documents.

However, you may reason, calculate, infer,
and derive answers from the information present
in the retrieved documents whenever sufficient
evidence exists.

Rules:

1. If the answer exists directly in the documents,
   answer from the documents.

2. If the documents contain an exercise, question,
   problem statement, activity, example, formula,
   theorem, rule, definition, or method that enables
   solving the user's question, derive the answer
   using that information.

3. You may perform calculations, logical reasoning,
   mathematical steps, or inference when necessary.

4. Do not invent facts that are unrelated to the documents.

5. If the documents provide enough information to solve
   the problem, solve it completely.

6. Return fallback ONLY when the documents are genuinely
   unrelated to the user question.


Rules:

1. If the answer can be reasonably inferred from the provided documents,
   answer the question.

2. If the documents contain a question, exercise, activity, definition,
   explanation, observation, or related information that directly supports
   the answer, use it.

3. Do NOT require the answer to be written verbatim.

4. Return the fallback ONLY when the documents are completely unrelated to
   the user question.

Exercise Solving Rules:

If the retrieved documents contain:

- exercises
- questions
- practice problems
- assignments
- activities

and the user is asking one of those questions,

then solve the question using the concepts,
methods, formulas, examples, or rules found in
the documents.

The final answer does NOT need to appear verbatim
inside the documents.


The retrieved documents may contain questions,
examples, procedures, formulas, rules, concepts,
instructions, reference material, or partial
information rather than explicit answers.

When enough information exists in the retrieved
documents to determine the answer, you should
reason from the provided content and generate
the final answer.

You are allowed to:

- perform calculations
- apply formulas
- execute described procedures
- follow algorithms
- infer conclusions
- solve exercises
- answer questions based on examples
- combine information from multiple retrieved chunks

Do not respond that the answer is unavailable
simply because the exact wording of the final
answer does not appear in the documents.

Only return an insufficiency response when the
retrieved documents genuinely lack the information
required to derive the answer.

Fallback:
{get_fallback_message(target_language)}

Language rules:
- Answer only in {language_name(target_language)}.
- Do not mix languages.
- Do not include URLs.
- Do not include References.

User question:
{query}

Documents:
{context}
"""
        try:
            client = genai.Client(api_key=self.GOOGLE_API_KEY)
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                ),
            )

            final_answer = (response.text or "").strip()
            print("=" * 80)
            print("RAW GEMINI ANSWER")
            print(final_answer)
            print("=" * 80)
            if final_answer.strip() == get_fallback_message(target_language):
                print("=" * 80)
                print("FALLBACK WITH DOCS")
                print("DOC COUNT:", len(docs))
                for i, doc in enumerate(docs[:3], start=1):
                    print(f"DOC {i} PAGE={doc.get('page_no')}")
                    print(doc.get("text", "")[:500])
                    print("-" * 80)
                print("=" * 80)
            if not final_answer:
                return get_fallback_message(target_language)

            if detect_query_language(final_answer, default=target_language) != target_language:
                final_answer = await self.translate_to_target(final_answer, target_language)

            return final_answer.strip() or get_fallback_message(target_language)
        except Exception as e:
            raise Exception(f"Error in gen-ai answer creation: {e}")

    async def query_search(
        self,
        query,
        user_id,
        category_id,
        client_id,
        index_name,
        answer_type,
        target_language="en",
        is_research=False,
    ):
        try:
            target_language = normalize_language_code(target_language or detect_query_language(query))
            
            print("=" * 80)
            print("BEFORE _retrieve_docs")
            print("QUERY:", query)
            print("ANSWER_TYPE VARIABLE:", answer_type)
            print("TYPE OF ANSWER_TYPE:", type(answer_type))
            print("=" * 80)
            docs = await self._retrieve_docs(query, user_id, category_id, index_name, answer_type)
            print(f"Retrieved {len(docs)} docs")
            print("=" * 80)
            print("DOCS SENT TO GEMINI")
            print("DOC COUNT:", len(docs))
            for i, d in enumerate(docs):
                print(
                    f"DOC {i+1} "
                    f"PAGE={d.get('page_no')} "
                    f"SCORE={d.get('score')}"
                )
                print(d.get("text", "")[:500])
                print("-" * 80)

            if is_research:
                return None, docs

            answer = await self.get_genai_answer(query, docs, answer_type, target_language)
            print("=" * 80)
            print("FINAL ANSWER RETURNED TO API/FRONTEND")
            print(answer)
            print("=" * 80)
            return answer, docs
        except Exception as e:
            raise Exception(f"Error in searching Query: {e}")

    async def research_query_search_with_subqueries(
        self,
        query,
        user_id,
        category_id,
        client_id,
        index_name,
        answer_type,
        sub_queries,
        target_language="en",
    ):
        try:
            target_language = normalize_language_code(target_language or detect_query_language(query))
            candidate_queries = sub_queries or []
            if not candidate_queries:
                candidate_queries = self.query_decomposer.generate_sub_queries(query)

            results = await asyncio.gather(
                *[
                    self.query_search(
                        sub_query,
                        user_id,
                        category_id,
                        client_id,
                        index_name,
                        answer_type,
                        target_language=target_language,
                        is_research=True,
                    )
                    for sub_query in candidate_queries
                ]
            )

            all_docs = []
            for sub_query, (_, docs) in zip(candidate_queries, results):
                for doc in docs:
                    all_docs.append({**doc, "sub_query": sub_query})

            seen = set()
            unique_docs = []
            for doc in all_docs:
                key = (doc.get("pdf_name", ""), doc.get("page_no", ""), doc.get("text", "")[:160])
                if key in seen:
                    continue
                seen.add(key)
                unique_docs.append(doc)

            final_answer = await self.get_genai_answer(
                query,
                unique_docs[:10],
                answer_type,
                target_language,
            )
            return final_answer, unique_docs[:10]
        except Exception as e:
            raise Exception(f"Error in Research Query Search With Subqueries: {e}")

    async def combine_answer_from_both_sources(self, question, ans1, ans2, target_language="en"):
        target_language = normalize_language_code(target_language or detect_query_language(question))

        def clean_answer(text: str) -> str:
            if not text:
                return ""
            text = re.sub(r"<[^>]+>", "", text)
            text = re.sub(r"\[.*?\]\(https?://[^\)]+\)", "", text)
            text = re.sub(r"https?://\S+", "", text)
            text = re.sub(r"###\s*(References|संदर्भ).*", "", text, flags=re.DOTALL | re.IGNORECASE)
            return re.sub(r"\s+", " ", text).strip()

        clean_ans1 = clean_answer(ans1)
        clean_ans2 = clean_answer(ans2)

        if not clean_ans1 and not clean_ans2:
            return get_fallback_message(target_language)
        if clean_ans1 and not clean_ans2:
            clean_ans2 = ""
        if clean_ans2 and not clean_ans1:
            clean_ans1 = ""

        try:
            prompt = COMBINE_ANSWER_PROMPT.format(
                user_query=question,
                internal_answer=clean_ans1,
                docbrains_answer=clean_ans2,
                target_language=language_name(target_language),
            )
            prompt += f"""

Final answer rules:
- Use only the two provided answers.
- Remove duplicates.
- Respond only in {language_name(target_language)}.
- Do not include URLs, HTML tags, citation numbers, or a References section.
- If neither answer contains useful content, return exactly: {get_fallback_message(target_language)}
"""
            llm = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                temperature=0,
                google_api_key=self.GOOGLE_API_KEY,
            )
            response = await llm.ainvoke(prompt)
            print("=" * 80)
            print("RAW GEMINI RESPONSE")
            print(response.content)
            print("=" * 80)

            final_answer = (response.content or "").strip()

            if detect_query_language(final_answer, default=target_language) != target_language:
                final_answer = await self.translate_to_target(final_answer, target_language)

            return final_answer or get_fallback_message(target_language)
        except Exception as e:
            raise Exception(f"Error combining answers: {e}")

    async def detect_query_language(self, user_query: str):
        return detect_query_language(user_query)
