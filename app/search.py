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
        WITH vector_search AS (
            SELECT
                c.id,
                ROW_NUMBER() OVER (
                    ORDER BY c.embedding <=> :embedding ::vector
                ) AS vector_rank
            FROM chunks c
            ORDER BY c.embedding <=> :embedding ::vector
            LIMIT 20
        ),
        fts_search AS (
            SELECT
                c.id,
                ROW_NUMBER() OVER (
                    ORDER BY ts_rank(
                        to_tsvector('english', c.content),
                        plainto_tsquery('english', :query)
                    ) DESC
                ) AS fts_rank
            FROM chunks c
            WHERE to_tsvector('english', c.content)
                @@ plainto_tsquery('english', :query)
            LIMIT 20
        ),
        rrf AS (
            SELECT
                COALESCE(vs.id, fs.id) AS id,
                COALESCE(1.0 / (60 + vs.vector_rank), 0) +
                COALESCE(1.0 / (60 + fs.fts_rank), 0) AS rrf_score
            FROM vector_search vs
            FULL OUTER JOIN fts_search fs ON vs.id = fs.id
        )
        SELECT
            c.id,
            c.content,
            c.page_number,
            c.token_count,
            d.filename,
            rrf.rrf_score
        FROM rrf
        JOIN chunks c ON c.id = rrf.id
        JOIN documents d ON d.id = c.document_id
        ORDER BY rrf.rrf_score DESC
        LIMIT :top_k
     """)

    result = db.execute(sql, {
        "embedding":  embedding_str,
        "query":      query,
        "top_k":      settings.TOP_K,
    })

    rows = result.fetchall()

    return [
        {
            "content":    row.content,
            "page_number": row.page_number,
            "filename":   row.filename,
            "rrf_score":  round(float(row.rrf_score), 4),
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