from pathlib import Path
import pandas as pd

from src.retrieval.faiss_index import VectorIndex
from src.retrieval.reranker import filter_and_rank


class VisualSearch:
    def __init__(self, articles_path: Path, embedding_metadata_path: Path, images_root: Path, index: VectorIndex, candidate_multiplier: int = 5):
        self.images_root = images_root
        self.index = index
        self.candidate_multiplier = candidate_multiplier
        self.articles = pd.read_csv(articles_path, dtype={"article_id": str})
        self.articles["article_id"] = self.articles["article_id"].str.zfill(10)
        self.articles = self.articles.set_index("article_id", drop=False)
        self.embedding_metadata = pd.read_csv(embedding_metadata_path, dtype={"article_id": str})
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pd.DataFrame:
        metadata = self.articles.reindex(self.embedding_metadata["article_id"])[[
            "article_id", "prod_name", "product_type_name", "product_group_name",
            "colour_group_name", "department_name", "section_name", "detail_desc",
        ]].copy()
        metadata["image_path"] = metadata["article_id"].map(
            lambda article_id: f"/images/{article_id[:3]}/{article_id}.jpg" # type: ignore
        )
        return metadata

    def search(self, query: list[float], limit: int = 20, filters: dict[str, str | None] | None = None) -> list[dict]:
        fetch_limit = min(max(limit * self.candidate_multiplier, limit), self.index.index.ntotal)
        scores, positions = self.index.search(query, fetch_limit)
        rows = []
        for score, position in zip(scores, positions):
            if position < 0 or position >= len(self.metadata):
                continue
            row = self.metadata.iloc[position].to_dict()
            row["similarity"] = round(float(score), 5)
            rows.append(row)
        candidates = pd.DataFrame(rows)
        if candidates.empty:
            return []
        return filter_and_rank(candidates, filters or {}, limit)
