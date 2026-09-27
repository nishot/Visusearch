from src.config import settings
from src.models.image_encoder import ImageEncoder
from src.retrieval.faiss_index import VectorIndex
from src.retrieval.search import VisualSearch


class VisualSearchService:
    def __init__(self):
        device = settings.resolved_device()
        self.encoder = ImageEncoder(settings.model_name, device)
        self.searcher = VisualSearch(
            settings.articles_path,
            settings.embedding_metadata_path,
            settings.images_root,
            VectorIndex(settings.index_path, settings.embeddings_path),
            settings.candidate_multiplier,
        )

    def search(self, payload: bytes, limit: int, filters: dict[str, str | None]):
        query = self.encoder.encode_bytes(payload)
        return self.searcher.search(query, limit, filters)
