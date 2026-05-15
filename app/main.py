from app.s3 import upload_to_s3, file_exists_in_s3
import os
import shutil
from uuid import UUID
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db, engine
from app.models import Base, Document, Chunk
from app.chunker import process_pdf
from app.embedder import embed_chunks
from app.search import search_chunks, generate_answer

# Create tables on startup if they don't exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG System")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload")
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Check if already in S3
    if file_exists_in_s3(file.filename):
        existing = db.query(Document).filter(Document.filename == file.filename).first()
        if existing:
            return {
                "message":     "Document already exists, skipping re-upload.",
                "document_id": str(existing.id),
                "filename":    existing.filename,
                "page_count":  existing.page_count,
            }

    # 2. Save file temporarily
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 3. Upload to S3
        s3_key = upload_to_s3(file_path, file.filename)

        # 4. Parse PDF into chunks
        chunks, page_count = process_pdf(file_path)

        # 5. Embed all chunks
        chunks = embed_chunks(chunks)

        # 6. Save document record
        document = Document(filename=file.filename, page_count=page_count, s3_key=s3_key)
        db.add(document)
        db.flush()

        # 7. Save all chunks
        for chunk in chunks:
            db.add(Chunk(
                document_id=document.id,
                content=chunk["content"],
                chunk_index=chunk["chunk_index"],
                page_number=chunk["page_number"],
                token_count=chunk["token_count"],
                embedding=chunk["embedding"],
            ))

        db.commit()

        return {
            "document_id": str(document.id),
            "filename":    file.filename,
            "page_count":  page_count,
            "chunks":      len(chunks),
            "s3_key":      s3_key,
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)



@app.post("/ask")
def ask(query: str, db: Session = Depends(get_db)):
    """
    Ask a question. Retrieve relevant chunks, generate GPT answer.
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Find relevant chunks
    chunks = search_chunks(query, db)

    # 2. Generate answer from chunks
    answer = generate_answer(query, chunks)

    return {
        "query":   query,
        "answer":  answer,
        "sources": chunks,
    }


@app.get("/documents")
def list_documents(db: Session = Depends(get_db)):
    """
    List all uploaded documents.
    """
    documents = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        {
            "document_id": str(doc.id),
            "filename":    doc.filename,
            "page_count":  doc.page_count,
            "created_at":  doc.created_at,
        }
        for doc in documents
    ]


@app.delete("/documents/{document_id}")
def delete_document(document_id: UUID, db: Session = Depends(get_db)):
    """
    Delete a document and all its chunks (CASCADE handles chunks automatically).
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    db.delete(doc)
    db.commit()

    return {"deleted": str(document_id)}             