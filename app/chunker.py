import tiktoken
from pypdf import PdfReader

# Tokenizer for text-embedding-3-small
tokenizer = tiktoken.get_encoding("cl100k_base")


def extract_text_from_pdf(file_path: str) -> list[dict]:
    """
    Read each page of the PDF and return a list of:
    { "page_number": int, "text": str }
    """
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page_number": i + 1, "text": text.strip()})
    return pages


def chunk_text(text: str, page_number: int, start_index: int) -> list[dict]:
    """
    Split a single page's text into token-aware chunks with overlap.
    Returns a list of chunk dicts.
    """
    from app.config import settings

    tokens = tokenizer.encode(text)
    chunks = []
    chunk_index = start_index
    start = 0

    while start < len(tokens):
        end = start + settings.CHUNK_SIZE
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "content":     chunk_text,
            "chunk_index": chunk_index,
            "page_number": page_number,
            "token_count": len(chunk_tokens),
        })

        chunk_index += 1
        start += settings.CHUNK_SIZE - settings.CHUNK_OVERLAP  # move forward with overlap

    return chunks


def process_pdf(file_path: str) -> tuple[list[dict], int]:
    """
    Full pipeline: PDF -> pages -> chunks.
    Returns (chunks, page_count)
    """
    pages = extract_text_from_pdf(file_path)
    all_chunks = []
    chunk_index = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], page["page_number"], chunk_index)
        all_chunks.extend(page_chunks)
        chunk_index += len(page_chunks)

    return all_chunks, len(pages)