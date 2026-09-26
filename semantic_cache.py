from sentence_transformers import SentenceTransformer
import numpy as np
import pickle


embedding_model = SentenceTransformer(
    "intfloat/multilingual-e5-base"
)


class SemanticCache:

    @staticmethod
    def embed(text):
        return embedding_model.encode(
            text,
            normalize_embeddings=True
        )

    @staticmethod
    def similarity(a, b):
        return float(np.dot(a, b))


def get_query_embedding(query: str):
    return embedding_model.encode(
        query,
        normalize_embeddings=True
    ).astype(np.float32)


async def semantic_cache_search(
    redis_client,
    query,
    answer_type,
    threshold=0.91,
):
    query_embedding = get_query_embedding(query)

    # Dynamic threshold based on answer type

    if answer_type in ["date", "location", "person", "number"]:
        threshold = 0.97

    elif answer_type in ["definition", "yes_no"]:
        threshold = 0.94

    else:
        threshold = 0.91

    print(
        f"SEMANTIC THRESHOLD={threshold} "
        f"ANSWER_TYPE={answer_type}"
    )

    best_answer = None
    best_score = 0
    best_query = None
    best_key = None

    async for key in redis_client.scan_iter(match="qa:*"):

        print("CHECKING KEY:", key)
        print("RAW KEY:", repr(key))
        if isinstance(key, bytes):
            key = key.decode()

        # skip embedding keys

        if (
            key.endswith(":embedding")
            or key.endswith(":query")
            or key.endswith(":answer_type")
        ):
            
            continue
        emb_key = f"{key}:embedding"

        print("LOOKING FOR:", emb_key)

        stored = await redis_client.get(emb_key)

        query_key = f"{key}:query"

        stored_query = await redis_client.get(query_key)

        if stored_query:
            if isinstance(stored_query, bytes):
                stored_query = stored_query.decode()

            print("STORED QUERY:", stored_query)

        print("FOUND:", stored is not None)

        if not stored:
            print("EMBEDDING NOT FOUND:", emb_key)
            continue

        stored_answer_type = await redis_client.get(
            f"{key}:answer_type"
        )

        if not stored_answer_type:
            print("NO ANSWER TYPE FOUND")
            continue

        if isinstance(stored_answer_type, bytes):
            stored_answer_type = stored_answer_type.decode()

        print(
            f"ANSWER TYPE CHECK: "
            f"stored={stored_answer_type}, "
            f"current={answer_type}"
        )

        
        if stored_answer_type != answer_type:
            print(
                f"SKIP TYPE MISMATCH "
                f"{stored_answer_type} != {answer_type}"
            )
            continue

        cached_embedding = pickle.loads(stored)

        score = SemanticCache.similarity(
            query_embedding,
            cached_embedding
        )

#        print(f"SCORE={score:.4f} KEY={key}")
        print(
            f"SCORE={score:.4f} "
            f"TYPE={stored_answer_type} "
            f"QUERY={stored_query}"
        )   
    
        if score < threshold:
            print(
            f"SKIP LOW SCORE "
            f"{score:.4f} < {threshold}"
        )
            continue
        
        if score > best_score:

            best_score = score

            best_query = stored_query
            best_key = key

            answer = await redis_client.get(key)

            best_answer = (
                answer.decode()
                if isinstance(answer, bytes)
                else answer
            )


    print(f"BEST SEMANTIC SCORE = {best_score}")

    if best_score >= threshold:

        print("=" * 80)
        print("SEMANTIC CACHE HIT")
        print("BEST MATCH QUERY:", best_query)
        print("CURRENT QUERY:", query)
        print("BEST SCORE:", best_score)
        print("CACHE KEY:", best_key)
        print("ANSWER:")
        print(best_answer)
        print("=" * 80)

        return best_answer
    return None



async def store_semantic_cache(
    redis_client,
    cache_key,
    query,
    answer_type
):

    print("STORING SEMANTIC CACHE:", cache_key)
    embedding = get_query_embedding(query)

    await redis_client.set(
        f"{cache_key}:embedding",
        pickle.dumps(embedding),
        ex=3600
    )

    await redis_client.set(
        f"{cache_key}:query",
        query,
        ex=3600
    )

    print(f"STORING ANSWER TYPE: {answer_type}")
    await redis_client.set(
        f"{cache_key}:answer_type",
        answer_type,
        ex=3600
    )

    print("WRITING:", f"{cache_key}:embedding")
    print("STORING SEMANTIC CACHE:", cache_key)
