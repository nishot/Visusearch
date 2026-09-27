from pathlib import Path

import numpy as np


class VectorIndex:
    def __init__(self, index_path: Path, embeddings_path: Path):
        import faiss
        self.faiss = faiss
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.index = self._load()

    def _load(self):
        if self.index_path.exists():
            return self.faiss.read_index(str(self.index_path))
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                "Search assets are missing. Run scripts/build_embeddings.py and scripts/build_index.py."
            )
        vectors = np.load(self.embeddings_path, mmap_mode="r").astype("float32")
        index = self.faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    def search(self, query: list[float], limit: int) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray([query], dtype="float32")
        scores, positions = self.index.search(vector, limit)
        return scores[0], positions[0]
