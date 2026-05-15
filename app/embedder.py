import hashlib
import json
import redis
import openai
from app.config import settings

# Initialize clients once at module level, not on every function call
redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)


def _cache_key(text: str) -> str:
    """
    Generate a unique Redis key for a piece of text.
    sha256 of the text means same text always gets same key.
    """
    hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    return f"emb:{settings.EMBEDDING_MODEL}:{hash}"


def embed_text(text: str) -> list[float]:
    """
    Embed a single piece of text.
    Checks Redis first — only calls OpenAI if not cached.
    """
    key = _cache_key(text)

    # 1. Check cache
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)

    # 2. Cache miss — call OpenAI
    response = openai_client.embeddings.create(
        model=settings.EMBEDDING_MODEL,
        input=text,
    )
    embedding = response.data[0].embedding

    # 3. Store in Redis with TTL
    redis_client.setex(key, settings.CACHE_TTL, json.dumps(embedding))

    return embedding


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunks in batches of 100.
    Adds 'embedding' key to each chunk dict.
    """
    BATCH_SIZE = 100

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        texts = [chunk["content"] for chunk in batch]

        # Check cache for each text individually
        embeddings = [embed_text(text) for text in texts]

        for chunk, embedding in zip(batch, embeddings):
            chunk["embedding"] = embedding

    return chunks