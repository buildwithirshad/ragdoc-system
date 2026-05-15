from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import OpenAI
from app.config import settings
from app.embedder import embed_text

openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)


def search_chunks(query: str, db: Session) -> list[dict]:
    """
    Embed the query, find the most similar chunks in the database.
    Returns top K chunks above the minimum similarity threshold.
    """
    query_embedding = embed_text(query)
    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    sql = text("""
        SELECT
            c.id,
            c.content,
            c.page_number,
            c.token_count,
            d.filename,
            1 - (c.embedding <=> :embedding ::vector) AS similarity
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE 1 - (c.embedding <=> :embedding ::vector) >= :min_similarity
        ORDER BY c.embedding <=> :embedding ::vector
        LIMIT :top_k
    """)

    result = db.execute(sql, {
        "embedding":     embedding_str,
        "min_similarity": settings.MIN_SIMILARITY,
        "top_k":         settings.TOP_K,
    })

    rows = result.fetchall()

    return [
        {
            "content":    row.content,
            "page_number": row.page_number,
            "filename":   row.filename,
            "similarity": round(float(row.similarity), 4),
        }
        for row in rows
    ]


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Take the retrieved chunks and ask GPT to answer the query using them.
    """
    if not chunks:
        return "I could not find any relevant information in the uploaded documents."

    # Build context from chunks
    context = "\n\n".join([
        f"[Page {c['page_number']} - {c['filename']}]\n{c['content']}"
        for c in chunks
    ])

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

Question: {query}

Answer:"""

    response = openai_client.chat.completions.create(
        model=settings.CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,       # 0 = deterministic, no hallucination
        max_tokens=500,
    )

    return response.choices[0].message.content.strip()