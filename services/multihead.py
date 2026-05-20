"""
Multi-head projection layer for DINOv2 embeddings.

Takes the 768-dim backbone embedding and projects it through 5 independent
MLP heads, each producing a 128-dim task-specific embedding:

    style_head     -- aesthetic identity (minimalist vs streetwear vs romantic)
    fit_head       -- silhouette/cut (oversized vs tailored vs bodycon)
    material_head  -- fabric/texture (wool vs denim vs silk)
    compat_head    -- co-outfit compatibility
    occasion_head  -- context appropriateness (work vs casual vs going-out)

Heads are initialized with random weights (Xavier) and can be trained later
with contrastive learning on Polyvore + weak supervision from Florence tags.
Even random-init projections provide useful compressed features for retrieval.
"""

import os
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

BACKBONE_DIM = 768
HEAD_HIDDEN_DIM = 256
HEAD_OUTPUT_DIM = 128

HEAD_NAMES = ["style", "fit", "material", "compat", "occasion"]

WEIGHTS_DIR = Path(os.getenv(
    "MULTIHEAD_WEIGHTS_DIR",
    str(Path(__file__).resolve().parent.parent / "models" / "multihead"),
))


class ProjectionHead:
    """
    Two-layer MLP projection head: Linear(768→256) → ReLU → LayerNorm → Linear(256→128) → L2 norm.
    
    Operates in numpy for inference (no torch dependency at serve time).
    Weights are stored as .npz files per head.
    """

    def __init__(self, name: str, w1: np.ndarray, b1: np.ndarray,
                 ln_gamma: np.ndarray, ln_beta: np.ndarray,
                 w2: np.ndarray, b2: np.ndarray):
        self.name = name
        self.w1 = w1  # (768, 256)
        self.b1 = b1  # (256,)
        self.ln_gamma = ln_gamma  # (256,)
        self.ln_beta = ln_beta    # (256,)
        self.w2 = w2  # (256, 128)
        self.b2 = b2  # (128,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Project backbone embedding(s) to head space.
        
        x: (768,) or (N, 768)
        returns: (128,) or (N, 128), L2-normalized
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]

        h = x @ self.w1 + self.b1
        h = np.maximum(h, 0)  # ReLU

        # LayerNorm
        mean = h.mean(axis=-1, keepdims=True)
        var = h.var(axis=-1, keepdims=True)
        h = (h - mean) / np.sqrt(var + 1e-5)
        h = h * self.ln_gamma + self.ln_beta

        out = h @ self.w2 + self.b2

        # L2 normalize
        norms = np.linalg.norm(out, axis=-1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        out = out / norms

        if single:
            return out[0]
        return out


def _xavier_init(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """Xavier uniform initialization."""
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def _init_random_head(name: str, seed: int) -> ProjectionHead:
    """Create a head with Xavier-initialized weights."""
    rng = np.random.default_rng(seed)

    w1 = _xavier_init(BACKBONE_DIM, HEAD_HIDDEN_DIM, rng)
    b1 = np.zeros(HEAD_HIDDEN_DIM, dtype=np.float32)
    ln_gamma = np.ones(HEAD_HIDDEN_DIM, dtype=np.float32)
    ln_beta = np.zeros(HEAD_HIDDEN_DIM, dtype=np.float32)
    w2 = _xavier_init(HEAD_HIDDEN_DIM, HEAD_OUTPUT_DIM, rng)
    b2 = np.zeros(HEAD_OUTPUT_DIM, dtype=np.float32)

    return ProjectionHead(name, w1, b1, ln_gamma, ln_beta, w2, b2)


def _load_head(name: str) -> ProjectionHead:
    """Load a trained head from disk, or initialize randomly if no weights found."""
    weights_path = WEIGHTS_DIR / f"{name}_head.npz"

    if weights_path.exists():
        logger.info("Loading trained %s_head from %s", name, weights_path)
        data = np.load(weights_path)
        return ProjectionHead(
            name=name,
            w1=data["w1"],
            b1=data["b1"],
            ln_gamma=data["ln_gamma"],
            ln_beta=data["ln_beta"],
            w2=data["w2"],
            b2=data["b2"],
        )

    seed = hash(name) % (2**31)
    logger.info("No trained weights for %s_head, using Xavier init (seed=%d)", name, seed)
    return _init_random_head(name, seed)


@lru_cache(maxsize=1)
def _load_all_heads() -> dict[str, ProjectionHead]:
    """Load or initialize all 5 projection heads."""
    heads = {}
    for name in HEAD_NAMES:
        heads[name] = _load_head(name)
    return heads


def compute_multihead_embeddings(backbone_embedding: np.ndarray) -> dict[str, np.ndarray]:
    """
    Project a single DINOv2 backbone embedding through all 5 heads.
    
    Args:
        backbone_embedding: (768,) L2-normalized DINOv2 CLS token
        
    Returns:
        dict mapping head name to (128,) L2-normalized embedding
    """
    heads = _load_all_heads()
    return {name: head.forward(backbone_embedding) for name, head in heads.items()}


def compute_multihead_embeddings_batch(backbone_embeddings: np.ndarray) -> dict[str, np.ndarray]:
    """
    Project a batch of DINOv2 backbone embeddings through all 5 heads.
    
    Args:
        backbone_embeddings: (N, 768) L2-normalized DINOv2 CLS tokens
        
    Returns:
        dict mapping head name to (N, 128) L2-normalized embeddings
    """
    heads = _load_all_heads()
    return {name: head.forward(backbone_embeddings) for name, head in heads.items()}


def save_head_weights(name: str, head: ProjectionHead):
    """Save trained head weights to disk."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f"{name}_head.npz"
    np.savez(
        path,
        w1=head.w1, b1=head.b1,
        ln_gamma=head.ln_gamma, ln_beta=head.ln_beta,
        w2=head.w2, b2=head.b2,
    )
    logger.info("Saved %s_head weights to %s", name, path)


def get_head_output_dim() -> int:
    return HEAD_OUTPUT_DIM


# ── Compatibility Scorer ──────────────────────────────────────────────────────


class CompatibilityScorer:
    """
    MLP that predicts pairwise compatibility from compat_head embeddings.
    
    Input: [item_a_compat_vec ; item_b_compat_vec] (256-dim concatenation)
    Output: compatibility score (0-1)
    
    Architecture: Linear(256,128) → ReLU → Linear(128,64) → ReLU → Linear(64,1) → Sigmoid
    """

    def __init__(self, w1: np.ndarray, b1: np.ndarray,
                 w2: np.ndarray, b2: np.ndarray,
                 w3: np.ndarray, b3: np.ndarray):
        self.w1 = w1  # (256, 128)
        self.b1 = b1  # (128,)
        self.w2 = w2  # (128, 64)
        self.b2 = b2  # (64,)
        self.w3 = w3  # (64, 1)
        self.b3 = b3  # (1,)

    def score(self, compat_a: np.ndarray, compat_b: np.ndarray) -> float:
        """
        Score compatibility between two items.
        
        Args:
            compat_a: (128,) compat_head embedding for item A
            compat_b: (128,) compat_head embedding for item B
            
        Returns:
            float between 0 and 1 (1 = highly compatible)
        """
        x = np.concatenate([compat_a, compat_b])
        h = x @ self.w1 + self.b1
        h = np.maximum(h, 0)
        h = h @ self.w2 + self.b2
        h = np.maximum(h, 0)
        logit = (h @ self.w3 + self.b3).item()
        return 1.0 / (1.0 + np.exp(-logit))

    def score_batch(self, compat_a: np.ndarray, compat_b: np.ndarray) -> np.ndarray:
        """
        Score compatibility for batches of pairs.
        
        Args:
            compat_a: (N, 128) compat_head embeddings for items A
            compat_b: (N, 128) compat_head embeddings for items B
            
        Returns:
            (N,) array of scores between 0 and 1
        """
        x = np.concatenate([compat_a, compat_b], axis=-1)  # (N, 256)
        h = x @ self.w1 + self.b1
        h = np.maximum(h, 0)
        h = h @ self.w2 + self.b2
        h = np.maximum(h, 0)
        logits = (h @ self.w3 + self.b3).squeeze(-1)
        return 1.0 / (1.0 + np.exp(-logits))

    def score_outfit(self, compat_embeddings: list[np.ndarray]) -> float:
        """
        Score an entire outfit by averaging all pairwise compatibility scores.
        
        Args:
            compat_embeddings: list of (128,) compat_head embeddings for each item
            
        Returns:
            float between 0 and 1 (average pairwise compatibility)
        """
        n = len(compat_embeddings)
        if n < 2:
            return 1.0

        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += self.score(compat_embeddings[i], compat_embeddings[j])
                count += 1

        return total / count


@lru_cache(maxsize=1)
def get_compatibility_scorer() -> CompatibilityScorer | None:
    """Load trained compatibility scorer, or return None if not available."""
    scorer_path = WEIGHTS_DIR / "compat_scorer.npz"
    if not scorer_path.exists():
        logger.info("No compatibility scorer weights found at %s", scorer_path)
        return None

    logger.info("Loading compatibility scorer from %s", scorer_path)
    data = np.load(scorer_path)
    return CompatibilityScorer(
        w1=data["w1"], b1=data["b1"],
        w2=data["w2"], b2=data["b2"],
        w3=data["w3"], b3=data["b3"],
    )
