"""
FashionCLIP 2.0 embedding service.

Replaces OpenAI text-embedding-3-small with patrickjohncyh/fashion-clip,
a CLIP model fine-tuned on 800K+ fashion products.

- Image encoder: photo → 512-dim vector  (replaces GPT-4o vision + OpenAI embed)
- Text encoder:  text  → 512-dim vector  (replaces OpenAI embed for queries)

Both encoders share the same vector space, so text queries find matching images
via cosine similarity — no LLM needed at query time.
"""

import io
import logging
from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast

logger = logging.getLogger(__name__)

MODEL_NAME = "patrickjohncyh/fashion-clip"
EMBEDDING_DIM = 512


class FashionCLIPService:
    """Lazy-loaded singleton for FashionCLIP inference."""

    def __init__(self):
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return

        logger.info("Loading FashionCLIP model: %s", MODEL_NAME)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = CLIPModel.from_pretrained(MODEL_NAME).to(self._device).eval()
        self._processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_NAME)
        logger.info("FashionCLIP loaded on %s", self._device)

    @torch.no_grad()
    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Encode a clothing image to a 512-dim vector."""
        self._load()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        emb = self._model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().tolist()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        """Encode a text string to a 512-dim vector."""
        self._load()
        inputs = self._tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self._device)
        emb = self._model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().tolist()

    @torch.no_grad()
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode multiple text strings."""
        self._load()
        if not texts:
            return []
        inputs = self._tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(self._device)
        emb = self._model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().tolist()

    @torch.no_grad()
    def embed_images(self, images_bytes: list[bytes]) -> list[list[float]]:
        """Batch-encode multiple images."""
        self._load()
        if not images_bytes:
            return []
        images = [Image.open(io.BytesIO(b)).convert("RGB") for b in images_bytes]
        inputs = self._processor(images=images, return_tensors="pt").to(self._device)
        emb = self._model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().tolist()

    @torch.no_grad()
    def zero_shot_classify(self, image_bytes: bytes, labels: list[str]) -> dict[str, float]:
        """
        Zero-shot classification: compare image against text labels.
        Returns {label: score} sorted by score descending.
        """
        self._load()
        image_emb = np.array(self.embed_image(image_bytes))
        label_embs = np.array(self.embed_texts(labels))

        similarities = label_embs @ image_emb
        scores = dict(zip(labels, similarities.tolist()))
        return dict(sorted(scores.items(), key=lambda x: -x[1]))


# Module-level singleton — loaded on first use
_service = FashionCLIPService()


def embed_image(image_bytes: bytes) -> list[float]:
    """Encode a clothing image to a 512-dim vector."""
    return _service.embed_image(image_bytes)


def embed_text(text: str) -> list[float]:
    """Encode text to a 512-dim vector in the same space as images."""
    return _service.embed_text(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-encode text strings."""
    return _service.embed_texts(texts)


def embed_images(images_bytes: list[bytes]) -> list[list[float]]:
    """Batch-encode images."""
    return _service.embed_images(images_bytes)


def zero_shot_classify(image_bytes: bytes, labels: list[str]) -> dict[str, float]:
    """Zero-shot classification of an image against text labels."""
    return _service.zero_shot_classify(image_bytes, labels)


def get_embedding_dim() -> int:
    """Return the embedding dimension (512 for FashionCLIP)."""
    return EMBEDDING_DIM
