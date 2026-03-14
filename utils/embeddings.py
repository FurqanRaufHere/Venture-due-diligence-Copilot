"""
utils/embeddings.py
────────────────────────────────────────────────────────────────
STEP 8 — Embedding Pipeline Setup

WHAT IS AN EMBEDDING?
  An embedding converts text into a list of numbers (a vector).
  Example: "We help manufacturers manage inventory" → [0.12, -0.45, 0.88, ...]
  
  Two similar texts produce vectors that are mathematically close.
  Two unrelated texts produce vectors that are far apart.
  This is how we compare a new startup against historical ones — 
  not by reading them, but by measuring the distance between their vectors.

WHY sentence-transformers INSTEAD OF OpenAI embeddings?
  - Completely FREE — no API calls, no cost, runs locally
  - Works offline — no internet needed after first download
  - Fast — runs on CPU, ~50ms per text
  - Good enough for startup similarity matching
  - Model: "all-MiniLM-L6-v2" — 384-dimensional vectors, excellent quality/speed tradeoff

HOW IT WORKS:
  embed_text("some text")  →  numpy array of shape (384,)
  embed_texts(["text1", "text2"])  →  numpy array of shape (2, 384)

  The numbers don't mean anything on their own.
  Their VALUE comes from comparing them with cosine similarity:
    similarity(A, B) = dot(A, B) / (|A| * |B|)
    Range: -1 (opposite) to 1 (identical)
    In practice: > 0.7 = very similar, < 0.3 = unrelated

FIRST RUN:
  The model (~90MB) downloads automatically on first call.
  After that it's cached locally — no re-download needed.
"""

import os
import logging
import numpy as np
from typing import List, Union

logger = logging.getLogger(__name__)

# Global model instance — loaded once, reused across all calls
# This avoids reloading the 90MB model on every request
_model = None
MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Load and cache the sentence-transformer model.
    Downloads on first call (~90MB), then loads from cache.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {MODEL_NAME} (first load may take 10–30 seconds)...")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
        logger.info(f"Embedding model loaded. Vector dimensions: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_text(text: str) -> np.ndarray:
    """
    Convert a single text string into a 384-dimensional embedding vector.
    
    Args:
        text: Any text string (startup description, pitch deck summary, etc.)
    
    Returns:
        numpy array of shape (384,) — normalized float32 values
    
    Example:
        vec = embed_text("AI-powered supply chain platform")
        # vec.shape == (384,)
        # vec.dtype == float32
    """
    if not text or not text.strip():
        # Return zero vector for empty input rather than crashing
        return np.zeros(384, dtype=np.float32)
    
    model = get_embedding_model()
    # normalize_embeddings=True ensures vectors have unit length
    # which makes cosine similarity = dot product (faster)
    embedding = model.encode(
        text.strip(),
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embedding.astype(np.float32)


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """
    Convert a list of texts into a matrix of embedding vectors.
    Batching makes this much faster than calling embed_text() in a loop.
    
    Args:
        texts:      List of text strings
        batch_size: How many texts to process at once (tune for memory)
    
    Returns:
        numpy array of shape (len(texts), 384)
    
    Example:
        matrix = embed_texts(["startup A desc", "startup B desc"])
        # matrix.shape == (2, 384)
    """
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    
    # Replace empty strings with a placeholder so batch stays aligned
    cleaned = [t.strip() if t and t.strip() else "unknown" for t in texts]
    
    model = get_embedding_model()
    embeddings = model.encode(
        cleaned,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=len(cleaned) > 50,  # show progress for large batches
    )
    return embeddings.astype(np.float32)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embedding vectors.
    Since we normalize on encode, this is just a dot product.
    
    Returns:
        float between -1 and 1 (in practice 0 to 1 for these embeddings)
        0.9+ = nearly identical
        0.7–0.9 = very similar
        0.5–0.7 = somewhat similar
        < 0.5 = different
    """
    return float(np.dot(vec_a, vec_b))


def test_embedding_pipeline():
    """
    Quick sanity check — call this to verify embeddings are working.
    Prints similarity scores between related and unrelated texts.
    """
    print("Testing embedding pipeline...")
    
    texts = [
        "AI-powered supply chain optimization for manufacturers",
        "Machine learning platform for inventory and logistics management",
        "Mobile app for restaurant food delivery and ordering",
    ]
    
    vecs = embed_texts(texts)
    print(f"Vector shape: {vecs.shape}")  # Should be (3, 384)
    
    sim_related = cosine_similarity(vecs[0], vecs[1])
    sim_unrelated = cosine_similarity(vecs[0], vecs[2])
    
    print(f"Similarity (supply chain vs logistics): {sim_related:.3f}  ← should be HIGH (>0.7)")
    print(f"Similarity (supply chain vs food app):  {sim_unrelated:.3f}  ← should be LOW (<0.5)")
    
    assert sim_related > sim_unrelated, "Embeddings not working correctly!"
    print("✅ Embedding pipeline working correctly.")
    return True


if __name__ == "__main__":
    test_embedding_pipeline()