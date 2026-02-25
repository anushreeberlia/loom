"""
FashionCLIP 2.0 embedding service using ONNX Runtime.

Runs patrickjohncyh/fashion-clip locally without PyTorch.
ONNX Runtime is ~50MB vs torch's ~2GB — suitable for Railway deployment.

- Image encoder: photo → 512-dim vector
- Text encoder:  text  → 512-dim vector
- Both share the same vector space for cross-modal search.

Model files are downloaded from HuggingFace Hub on first use (~605MB, cached).
"""

import io
import os
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

MODEL_NAME = "patrickjohncyh/fashion-clip"
EMBEDDING_DIM = 512


class FashionCLIPService:
    """Lazy-loaded singleton for FashionCLIP ONNX inference."""

    def __init__(self):
        self._session = None
        self._processor = None
        self._tokenizer = None
        self._input_names = None
        self._output_names = None

    def _load(self):
        if self._session is not None:
            return

        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import CLIPProcessor, CLIPTokenizerFast

        logger.info("Downloading FashionCLIP ONNX model...")
        model_path = hf_hub_download(MODEL_NAME, "onnx/model.onnx")

        logger.info("Loading ONNX session...")
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_names = [out.name for out in self._session.get_outputs()]

        # Load from repo root; patrickjohncyh/fashion-clip has no onnx subfolder for processor/tokenizer
        self._processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        self._tokenizer = CLIPTokenizerFast.from_pretrained(MODEL_NAME)

        logger.info(
            "FashionCLIP loaded (ONNX). Inputs: %s, Outputs: %s",
            self._input_names, self._output_names
        )

    def _normalize(self, vec: np.ndarray) -> list[float]:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def _get_dummy_text_inputs(self) -> dict:
        """Create minimal dummy text inputs when only encoding images."""
        self._load()
        inputs = self._tokenizer("", return_tensors="np", padding="max_length", max_length=77)
        return {k: v for k, v in inputs.items() if k in self._input_names}

    def _get_dummy_image_inputs(self) -> dict:
        """Create minimal dummy image inputs when only encoding text."""
        self._load()
        dummy_image = Image.new("RGB", (224, 224), (128, 128, 128))
        inputs = self._processor(images=dummy_image, return_tensors="np")
        return {k: v for k, v in inputs.items() if k in self._input_names}

    def embed_image(self, image_bytes: bytes) -> list[float]:
        """Encode a clothing image to a 512-dim vector."""
        self._load()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_inputs = self._processor(images=image, return_tensors="np")

        feed = {k: v for k, v in image_inputs.items() if k in self._input_names}
        # CLIP ONNX may require both modalities; provide dummy text if needed
        if "input_ids" in self._input_names and "input_ids" not in feed:
            feed.update(self._get_dummy_text_inputs())

        outputs = self._session.run(None, feed)
        output_map = dict(zip(self._output_names, outputs))

        # Prefer projected image embeddings
        for key in ["image_embeds", "image_features", "image_embed"]:
            if key in output_map:
                return self._normalize(output_map[key][0])

        # Fallback: last output that's 512-dim
        for out in reversed(outputs):
            if out.ndim >= 1 and out.shape[-1] == EMBEDDING_DIM:
                vec = out[0] if out.ndim > 1 else out
                return self._normalize(vec)

        raise RuntimeError(f"Could not find {EMBEDDING_DIM}-dim image embedding in outputs: {self._output_names}")

    def embed_text(self, text: str) -> list[float]:
        """Encode text to a 512-dim vector in the same space as images."""
        self._load()
        text_inputs = self._tokenizer(
            text, return_tensors="np", padding=True, truncation=True, max_length=77
        )

        feed = {k: v for k, v in text_inputs.items() if k in self._input_names}
        if "pixel_values" in self._input_names and "pixel_values" not in feed:
            feed.update(self._get_dummy_image_inputs())

        outputs = self._session.run(None, feed)
        output_map = dict(zip(self._output_names, outputs))

        for key in ["text_embeds", "text_features", "text_embed"]:
            if key in output_map:
                return self._normalize(output_map[key][0])

        for out in reversed(outputs):
            if out.ndim >= 1 and out.shape[-1] == EMBEDDING_DIM:
                vec = out[0] if out.ndim > 1 else out
                return self._normalize(vec)

        raise RuntimeError(f"Could not find {EMBEDDING_DIM}-dim text embedding in outputs: {self._output_names}")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Batch-encode multiple text strings."""
        if not texts:
            return []
        # ONNX doesn't batch as cleanly as PyTorch, encode individually
        return [self.embed_text(t) for t in texts]

    def embed_images(self, images_bytes: list[bytes]) -> list[list[float]]:
        """Batch-encode multiple images."""
        if not images_bytes:
            return []
        return [self.embed_image(b) for b in images_bytes]

    def zero_shot_classify(self, image_bytes: bytes, labels: list[str]) -> dict[str, float]:
        """Compare image against text labels. Returns {label: score} sorted descending."""
        image_emb = np.array(self.embed_image(image_bytes))
        label_embs = np.array(self.embed_texts(labels))
        similarities = label_embs @ image_emb
        scores = dict(zip(labels, similarities.tolist()))
        return dict(sorted(scores.items(), key=lambda x: -x[1]))


# Module-level singleton
_service = FashionCLIPService()


def embed_image(image_bytes: bytes) -> list[float]:
    return _service.embed_image(image_bytes)


def embed_text(text: str) -> list[float]:
    return _service.embed_text(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _service.embed_texts(texts)


def embed_images(images_bytes: list[bytes]) -> list[list[float]]:
    return _service.embed_images(images_bytes)


def zero_shot_classify(image_bytes: bytes, labels: list[str]) -> dict[str, float]:
    return _service.zero_shot_classify(image_bytes, labels)


def get_embedding_dim() -> int:
    return EMBEDDING_DIM
