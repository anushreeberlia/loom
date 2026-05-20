"""
Evaluate trained multi-head projection layers.

Metrics:
1. Retrieval Recall@K: Given an anchor item, are co-outfit items in the top-K neighbors?
2. Compatibility scorer accuracy: Can the scorer distinguish real vs random outfits?
3. Embedding space quality: Visualization via t-SNE/UMAP colored by category/occasion.

Usage:
    python train/evaluate_heads.py --config train/config.yaml --head compat
    python train/evaluate_heads.py --config train/config.yaml --head all --visualize
"""

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_embeddings_and_project(h5_path: str, head_name: str, weights_dir: str):
    """Load backbone embeddings and project through a trained head."""
    import h5py
    from services.multihead import ProjectionHead

    # Load backbone embeddings
    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        item_ids = [x.decode() if isinstance(x, bytes) else x for x in f["item_ids"][:]]

    # Load trained head
    weights_path = Path(weights_dir) / f"{head_name}_head.npz"
    if not weights_path.exists():
        logger.error("No trained weights found at %s", weights_path)
        sys.exit(1)

    data = np.load(weights_path)
    head = ProjectionHead(
        name=head_name,
        w1=data["w1"], b1=data["b1"],
        ln_gamma=data["ln_gamma"], ln_beta=data["ln_beta"],
        w2=data["w2"], b2=data["b2"],
    )

    # Project all embeddings
    projected = head.forward(embeddings)
    logger.info("Projected %d items through %s_head: (%d,) → (%d,)",
                len(item_ids), head_name, embeddings.shape[1], projected.shape[1])

    return projected, item_ids


def compute_recall_at_k(
    projected: np.ndarray,
    item_ids: list[str],
    pairs_path: str,
    k_values: list[int] = None,
) -> dict:
    """
    Compute Recall@K for retrieval.
    
    For each anchor in the test pairs, find its K nearest neighbors in the
    projected space. Check if the true positive is among them.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    # Build ID to index mapping
    id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    # Load positive pairs
    positive_pairs = []
    with open(pairs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["label"]) == 1:
                if row["item_a"] in id_to_idx and row["item_b"] in id_to_idx:
                    positive_pairs.append((row["item_a"], row["item_b"]))

    # Use last 10% as test set
    n_test = max(len(positive_pairs) // 10, 100)
    test_pairs = positive_pairs[-n_test:]
    logger.info("Evaluating on %d test pairs", len(test_pairs))

    # Pre-compute similarity matrix (or do it per-query for large datasets)
    # For efficiency: batch cosine similarity
    norms = np.linalg.norm(projected, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = projected / norms

    recalls = {k: 0 for k in k_values}
    mrr_sum = 0.0

    for anchor_id, positive_id in test_pairs:
        anchor_idx = id_to_idx[anchor_id]
        positive_idx = id_to_idx[positive_id]

        # Cosine similarity to all items
        similarities = normalized[anchor_idx] @ normalized.T
        similarities[anchor_idx] = -1  # exclude self

        # Rank by similarity
        ranked_indices = np.argsort(-similarities)

        # Find rank of positive
        rank = np.where(ranked_indices == positive_idx)[0][0] + 1  # 1-indexed

        mrr_sum += 1.0 / rank

        for k in k_values:
            if rank <= k:
                recalls[k] += 1

    n = len(test_pairs)
    results = {f"recall@{k}": recalls[k] / n for k in k_values}
    results["mrr"] = mrr_sum / n
    results["n_test_pairs"] = n

    return results


def compute_scorer_accuracy(projected: np.ndarray, item_ids: list[str], pairs_path: str, weights_dir: str) -> dict:
    """Evaluate the compatibility scorer on held-out pairs."""
    scorer_path = Path(weights_dir) / "compat_scorer.npz"
    if not scorer_path.exists():
        return {"scorer_accuracy": None, "note": "no scorer weights found"}

    data = np.load(scorer_path)
    id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}

    # Load all pairs (positive + negative)
    all_pairs = []
    with open(pairs_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["item_a"] in id_to_idx and row["item_b"] in id_to_idx:
                all_pairs.append((row["item_a"], row["item_b"], int(row["label"])))

    # Use last 10% as test
    n_test = max(len(all_pairs) // 10, 200)
    test_pairs = all_pairs[-n_test:]

    # Score using numpy scorer
    correct = 0
    for item_a, item_b, label in test_pairs:
        emb_a = projected[id_to_idx[item_a]]
        emb_b = projected[id_to_idx[item_b]]
        concat = np.concatenate([emb_a, emb_b])

        # Forward pass through scorer
        h = concat @ data["w1"] + data["b1"]
        h = np.maximum(h, 0)
        h = h @ data["w2"] + data["b2"]
        h = np.maximum(h, 0)
        score = 1.0 / (1.0 + np.exp(-(h @ data["w3"] + data["b3"])))
        score = score.item()

        predicted = 1 if score > 0.5 else 0
        if predicted == label:
            correct += 1

    accuracy = correct / len(test_pairs)
    return {"scorer_accuracy": accuracy, "n_test": len(test_pairs)}


def visualize_embeddings(projected: np.ndarray, item_ids: list[str], head_name: str, output_dir: Path):
    """Generate t-SNE visualization of the embedding space."""
    try:
        from sklearn.manifold import TSNE
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("sklearn/matplotlib not installed, skipping visualization")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Subsample for visualization (t-SNE is O(n^2))
    max_points = 5000
    if len(projected) > max_points:
        indices = np.random.choice(len(projected), max_points, replace=False)
        vis_embeddings = projected[indices]
    else:
        vis_embeddings = projected
        indices = np.arange(len(projected))

    logger.info("Running t-SNE on %d points...", len(vis_embeddings))
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(vis_embeddings)

    plt.figure(figsize=(12, 8))
    plt.scatter(coords[:, 0], coords[:, 1], s=2, alpha=0.5)
    plt.title(f"{head_name}_head embedding space (t-SNE)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.tight_layout()

    output_path = output_dir / f"{head_name}_tsne.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Saved visualization to %s", output_path)


def compare_vs_random(h5_path: str, pairs_path: str, head_name: str):
    """Compare trained head retrieval vs random-init head."""
    import h5py
    from services.multihead import _init_random_head

    with h5py.File(h5_path, "r") as f:
        embeddings = f["embeddings"][:]
        item_ids = [x.decode() if isinstance(x, bytes) else x for x in f["item_ids"][:]]

    # Random head projection
    seed = hash(head_name) % (2**31)
    random_head = _init_random_head(head_name, seed)
    random_projected = random_head.forward(embeddings)

    # Compute recall with random head
    random_results = compute_recall_at_k(random_projected, item_ids, pairs_path)
    return {"random_init": random_results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained multi-head projections")
    parser.add_argument("--config", type=Path, default=Path("train/config.yaml"))
    parser.add_argument("--head", type=str, default="compat",
                       choices=["all", "compat", "style", "occasion", "fit", "material"])
    parser.add_argument("--visualize", action="store_true", help="Generate t-SNE plots")
    parser.add_argument("--compare-random", action="store_true", help="Compare vs random init baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    h5_path = config["data"]["backbone_embeddings"]
    weights_dir = config["output"]["weights_dir"]

    HEAD_TO_PAIRS = {
        "compat": config["data"]["compat_pairs"],
        "style": config["data"]["style_pairs"],
        "occasion": config["data"].get("occasion_pairs", ""),
        "fit": config["data"].get("fit_pairs", ""),
        "material": config["data"].get("material_pairs", ""),
    }

    heads_to_eval = list(HEAD_TO_PAIRS.keys()) if args.head == "all" else [args.head]

    for head_name in heads_to_eval:
        pairs_path = HEAD_TO_PAIRS.get(head_name, "")
        if not pairs_path or not Path(pairs_path).exists():
            logger.warning("No pairs file for %s_head, skipping", head_name)
            continue

        logger.info("\n" + "=" * 60)
        logger.info("Evaluating %s_head", head_name)
        logger.info("=" * 60)

        projected, item_ids = load_embeddings_and_project(h5_path, head_name, weights_dir)

        # Recall@K
        recall_results = compute_recall_at_k(projected, item_ids, pairs_path)
        logger.info("Retrieval metrics:")
        for metric, value in recall_results.items():
            if isinstance(value, float):
                logger.info("  %s: %.4f", metric, value)
            else:
                logger.info("  %s: %s", metric, value)

        # Scorer accuracy (compat only)
        if head_name == "compat":
            scorer_results = compute_scorer_accuracy(projected, item_ids, pairs_path, weights_dir)
            logger.info("Scorer metrics:")
            for metric, value in scorer_results.items():
                logger.info("  %s: %s", metric, value)

        # Compare vs random baseline
        if args.compare_random:
            random_results = compare_vs_random(h5_path, pairs_path, head_name)
            logger.info("Random init baseline:")
            for metric, value in random_results["random_init"].items():
                if isinstance(value, float):
                    logger.info("  %s: %.4f", metric, value)

        # Visualization
        if args.visualize:
            visualize_embeddings(projected, item_ids, head_name, Path("train/plots"))


if __name__ == "__main__":
    main()
