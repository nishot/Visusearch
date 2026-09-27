from io import BytesIO
from threading import Lock

import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


class ImageEncoder:
    def __init__(self, model_name: str, device: str):
        self.device = torch.device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self._lock = Lock()

    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._lock, torch.no_grad():
            output = self.model.get_image_features(**inputs)
        if isinstance(output, torch.Tensor):
            vectors = output
        elif hasattr(output, "pooler_output"):
            vectors = output.pooler_output
        else:
            vectors = output[1]
        return torch.nn.functional.normalize(vectors, p=2, dim=1).cpu()

    def encode_bytes(self, payload: bytes) -> list[float]:
        try:
            image = Image.open(BytesIO(payload)).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("The uploaded file is not a valid image.") from exc
        return self.encode_images([image])[0].numpy().tolist()
