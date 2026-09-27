from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    project_root: Path = Path(__file__).resolve().parents[1]
    dataset_root: Path = project_root / "dataset" / "h-and-m-personalized-fashion-recommendations"
    articles_path: Path = dataset_root / "articles.csv"
    images_root: Path = dataset_root / "images"
    embeddings_path: Path = project_root / "embeddings" / "image_embeddings.npy"
    embedding_metadata_path: Path = project_root / "embeddings" / "embedding_metadata.csv"
    index_path: Path = project_root / "indexes" / "fashion_images.faiss"
    model_name: str = os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32")
    device: str = os.getenv("CLIP_DEVICE", "auto")
    candidate_multiplier: int = int(os.getenv("SEARCH_CANDIDATE_MULTIPLIER", "5"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


settings = Settings()
