"""
vector_search.py — the "find the relevant part of the report" module
----------------------------------------------------------------------
What's a vector embedding?
  It's a way to turn a piece of text into a list of numbers (a "vector")
  that captures its MEANING. Text with similar meaning ends up with
  similar numbers. So "hemoglobin is low" and "your iron levels are
  reduced" would produce vectors that are close to each other, even
  though the words are different.

What's FAISS?
  It's a library that takes a big pile of these number-vectors and lets
  you quickly ask: "which of these is closest in meaning to MY vector?"
  That's how we find the part of the report most relevant to your question,
  instead of stuffing the entire report into every single message.

The flow:
  1. Split the report into small chunks (a few sentences each).
  2. Turn each chunk into a vector (embed it).
  3. Store all the vectors in a FAISS index.
  4. When you ask a question, turn the question into a vector too, and
     ask FAISS which chunks are the closest match.
  5. Send just those chunks (not the whole report) to the AI as context.
"""

import faiss
from sentence_transformers import SentenceTransformer

# This is a small, free, local embedding model (~80MB, downloads once).
_model = None


def get_model():
    """Load the embedding model once and reuse it (loading it is slow)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """
    Split text into overlapping chunks of `chunk_size` characters.
    Overlap helps avoid cutting a sentence in half between two chunks.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def build_index(chunks: list):
    """
    Turn a list of text chunks into a searchable FAISS index.
    Returns (index, chunks) — we keep the original chunks around so that
    once FAISS tells us "chunk #3 is the best match", we can look up its text.
    """
    model = get_model()
    vectors = model.encode(chunks)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)  # simplest FAISS index: plain distance search
    index.add(vectors)
    return index, chunks


def search(index, chunks: list, query: str, top_k: int = 3) -> list:
    """Return the top_k chunks most relevant to the query."""
    model = get_model()
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]
