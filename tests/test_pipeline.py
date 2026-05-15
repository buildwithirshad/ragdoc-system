import pytest
from unittest.mock import patch, MagicMock
from app.chunker import chunk_text, extract_text_from_pdf
from app.embedder import embed_text, _cache_key
from app.s3 import upload_to_s3, file_exists_in_s3


# chunker test
def test_chunk_text_basic():
    """A short text should produce exactly one chunk."""
    text = "This is a short sentence."
    chunks = chunk_text(text, page_number=1, start_index=0)

    assert len(chunks) == 1
    assert chunks[0]["page_number"] == 1
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["token_count"] > 0
    assert chunks[0]["content"] == text


def test_chunk_text_overlap():
    """A long text should produce multiple chunks with correct index increment."""
    # Generate a long text that exceeds CHUNK_SIZE
    text = "word " * 600  # ~600 tokens
    chunks = chunk_text(text, page_number=2, start_index=0)

    assert len(chunks) > 1
    # chunk_index should increment by 1 each time
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i
        assert chunk["page_number"] == 2


def test_chunk_text_empty():
    """Empty text should return no chunks."""
    chunks = chunk_text("", page_number=1, start_index=0)
    assert chunks == []


# embedder tests
def test_cache_key_consistent():
    """Same text should always produce the same cache key."""
    key1 = _cache_key("hello world")
    key2 = _cache_key("hello world")
    assert key1 == key2


def test_cache_key_different():
    """Different text should produce different cache keys."""
    key1 = _cache_key("hello world")
    key2 = _cache_key("goodbye world")
    assert key1 != key2


def test_embed_text_uses_cache():
    """If embedding is in Redis, OpenAI should NOT be called."""
    fake_embedding = [0.1] * 1536

    with patch("app.embedder.redis_client") as mock_redis, \
         patch("app.embedder.openai_client") as mock_openai:

        # Simulate cache hit
        mock_redis.get.return_value = str(fake_embedding)

        import json
        mock_redis.get.return_value = json.dumps(fake_embedding)

        result = embed_text("some text")

        # Redis was checked
        assert mock_redis.get.called
        # OpenAI was NOT called
        assert not mock_openai.embeddings.create.called
        # Correct embedding returned
        assert result == fake_embedding


def test_embed_text_calls_openai_on_cache_miss():
    """If embedding is not in Redis, OpenAI should be called and result cached."""
    fake_embedding = [0.2] * 1536

    with patch("app.embedder.redis_client") as mock_redis, \
         patch("app.embedder.openai_client") as mock_openai:

        # Simulate cache miss
        mock_redis.get.return_value = None

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.data[0].embedding = fake_embedding
        mock_openai.embeddings.create.return_value = mock_response

        result = embed_text("some text")

        # OpenAI was called
        assert mock_openai.embeddings.create.called
        # Result was stored in Redis
        assert mock_redis.setex.called
        # Correct embedding returned
        assert result == fake_embedding

# S3 tests
def test_file_exists_in_s3_returns_true():
    """If S3 head_object succeeds, file exists."""
    with patch("app.s3.s3_client") as mock_s3:
        mock_s3.head_object.return_value = {}
        result = file_exists_in_s3("test.pdf")
        assert result is True


def test_file_exists_in_s3_returns_false():
    """If S3 head_object raises ClientError, file does not exist."""
    from botocore.exceptions import ClientError
    with patch("app.s3.s3_client") as mock_s3:
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
        )
        result = file_exists_in_s3("test.pdf")
        assert result is False


def test_upload_to_s3_returns_correct_key():
    """Upload should return the correct S3 key."""
    with patch("app.s3.s3_client") as mock_s3:
        mock_s3.upload_file.return_value = None
        key = upload_to_s3("/tmp/test.pdf", "test.pdf")
        assert key == "documents/test.pdf"