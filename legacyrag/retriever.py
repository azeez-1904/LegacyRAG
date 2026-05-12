"""
Vector retriever for LegacyRAG.

In-memory cosine similarity retrieval over embedded document chunks.
Persists the store to disk as a .npz + JSON pair so ingested documents
survive server restarts.
"""

import json
import logging
import os
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)

STORE_EMBEDDINGS_FILE = "vector_store.npz"
STORE_METADATA_FILE = "vector_store_meta.json"


@dataclass
class Chunk:
    doc_id: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore:
    """Flat in-memory cosine similarity store with disk persistence."""

    def __init__(self) -> None:
        self._embeddings: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Append chunks and their embeddings to the store."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({embeddings.shape[0]}) must match"
            )

        normed = _l2_normalize(embeddings)

        if self._embeddings.size == 0:
            self._embeddings = normed
        else:
            self._embeddings = np.vstack([self._embeddings, normed])

        self._chunks.extend(chunks)
        logger.info("VectorStore: added %d chunks (total %d)", len(chunks), len(self._chunks))

    def clear(self) -> None:
        self._embeddings = np.empty((0, 0), dtype=np.float32)
        self._chunks = []

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        """Return top_k (chunk, score) pairs by cosine similarity."""
        if len(self._chunks) == 0:
            return []

        q = _l2_normalize(query_embedding.reshape(1, -1))  # (1, D)
        scores: np.ndarray = (self._embeddings @ q.T).squeeze(axis=1)  # (N,)

        k = min(top_k, len(self._chunks))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._chunks[i], float(scores[i])) for i in top_indices]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        embeddings_path: str = STORE_EMBEDDINGS_FILE,
        meta_path: str = STORE_METADATA_FILE,
    ) -> None:
        np.savez_compressed(embeddings_path, embeddings=self._embeddings)
        with open(meta_path, "w") as fh:
            json.dump([asdict(c) for c in self._chunks], fh, indent=2)
        logger.info("VectorStore saved (%d chunks)", len(self._chunks))

    def load(
        self,
        embeddings_path: str = STORE_EMBEDDINGS_FILE,
        meta_path: str = STORE_METADATA_FILE,
    ) -> bool:
        """Load from disk. Returns True if data was found and loaded."""
        if not (os.path.exists(embeddings_path) and os.path.exists(meta_path)):
            return False
        try:
            data = np.load(embeddings_path)
            self._embeddings = data["embeddings"].astype(np.float32)
            with open(meta_path) as fh:
                raw = json.load(fh)
            self._chunks = [
                Chunk(
                    doc_id=c["doc_id"],
                    chunk_index=c["chunk_index"],
                    text=c["text"],
                    metadata=c.get("metadata", {}),
                )
                for c in raw
            ]
            logger.info("VectorStore loaded (%d chunks)", len(self._chunks))
            return True
        except Exception as exc:
            logger.error("VectorStore load failed: %s", exc)
            return False

    @property
    def size(self) -> int:
        return len(self._chunks)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    doc_id: str = "doc",
) -> list[Chunk]:
    """
    Split text into overlapping word-boundary chunks.

    chunk_size and overlap are measured in characters.
    """
    chunks: list[Chunk] = []
    start = 0
    index = 0
    text = text.strip()

    while start < len(text):
        end = start + chunk_size

        # Snap to word boundary unless we're at EOF
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(Chunk(doc_id=doc_id, chunk_index=index, text=chunk))
            index += 1

        start = max(start + 1, end - overlap)

    return chunks


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms
