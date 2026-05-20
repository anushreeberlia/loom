"""
DINOv2 ViT-B/14 visual backbone encoder.

Produces 768-dim CLS token embeddings from garment images.
Uses the frozen DINOv2-base model (86M params) via torch hub with lazy loading.

This is the shared backbone for all multi-head projection layers.
"""

import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 768
MODEL_NAME = "dinov2_vitb14"


@lru_cache(maxsize=1)
def _load_model():
    """Lazy-load DINOv2 model and preprocessing transform."""
    import torch

    logger.info("Loading DINOv2 ViT-B/14 model...")
    model = torch.hub.load("facebookresearch/dinov2", MODEL_NAME)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info("DINOv2 loaded on %s", device)

    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, transform, device


def _prepare_image(image_bytes: bytes, transform) -> "torch.Tensor":
    """Load image bytes into a preprocessed tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(img)


def embed_image(image_bytes: bytes) -> np.ndarray:
    """
    Encode a single image to a 768-dim DINOv2 CLS embedding.

    Returns L2-normalized numpy array (768,).
    """
    import torch

    model, transform, device = _load_model()
    tensor = _prepare_image(image_bytes, transform).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(tensor)

    emb = embedding[0].cpu().numpy()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb


def embed_images(image_bytes_list: list[bytes], batch_size: int = 16) -> np.ndarray:
    """
    Batch-encode multiple images to 768-dim DINOv2 CLS embeddings.

    Returns L2-normalized numpy array of shape (N, 768).
    """
    import torch

    model, transform, device = _load_model()

    all_embeddings = []
    for i in range(0, len(image_bytes_list), batch_size):
        batch_bytes = image_bytes_list[i:i + batch_size]
        tensors = [_prepare_image(b, transform) for b in batch_bytes]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            embeddings = model(batch)

        embs = embeddings.cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embs = embs / norms
        all_embeddings.append(embs)

    return np.concatenate(all_embeddings, axis=0)


def get_embedding_dim() -> int:
    return EMBEDDING_DIM
