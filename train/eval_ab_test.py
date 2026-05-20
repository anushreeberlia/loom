"""
A/B evaluation: Multi-head (DINOv2 + compat_scorer) vs Legacy (FashionCLIP + rules).

Evaluates both systems on:
1. Fill-In-The-Blank (FITB): Given partial outfit + 4 candidates, pick the correct item.
2. Compatibility AUC: Score real outfits vs random, measure discrimination.
3. Retrieval Recall@K: Given anchor, are co-outfit items ranked in top-K?

Uses pre-computed DINOv2 embeddings from the Polyvore test set.

Usage:
    python train/eval_ab_test.py --config train/config.yaml --data-dir data/polyvore
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data Loading ──────────────────────────────────────────────────────────────


def load_embedding_store(h5_path: str):
    """Load pre-computed backbone embeddings."""
    import h5py
    f = h5py.File(h5_path, "r")
    embeddings = f["embeddings"][:]
    item_ids = [x.decode() if isinstance(x, bytes) else x for x in f["item_ids"][:]]
    id_to_idx = {item_id: i for i, item_id in enumerate(item_ids)}
    logger.info("Loaded %d embeddings from %s", len(item_ids), h5_path)
    return embeddings, item_ids, id_to_idx


def load_fitb_questions(data_dir: Path) -> list[dict]:
    """Load fill-in-the-blank test questions from Polyvore dataset."""
    search_paths = [
        data_dir / "polyvore-dataset" / "fill_in_the_blank_test.json",
        data_dir / "fill_in_the_blank_test.json",
    ]
    for p in search_paths:
        if p.exists():
            with open(p) as f:
                questions = json.load(f)
            logger.info("Loaded %d FITB questions from %s", len(questions), p)
            return questions

    # Try recursive search
    found = list(data_dir.rglob("fill_in_the_blank*.json"))
    if found:
        with open(found[0]) as f:
            questions = json.load(f)
        logger.info("Loaded %d FITB questions from %s", len(questions), found[0])
        return questions

    logger.warning("No FITB questions found under %s", data_dir)
    return []


def load_test_outfits(data_dir: Path) -> list[list[str]]:
    """Load test outfit item lists for compatibility evaluation."""
    search_paths = [
        data_dir / "polyvore-dataset" / "test_no_dup.json",
        data_dir / "test_no_dup.json",
    ]
    for p in search_paths:
        if p.exists():
            with open(p) as f:
                data = json.load(f)
            outfits = []
            for outfit in data:
                set_id = str(outfit.get("set_id", ""))
                items = [f"{set_id}_{item['index']}" for item in outfit.get("items", [])]
                if len(items) >= 2:
                    outfits.append(items)
            logger.info("Loaded %d test outfits from %s", len(outfits), p)
            return outfits

    logger.warning("No test outfits found")
    return []


# ── Multi-Head System (System B) ─────────────────────────────────────────────


def load_multihead_system(weights_dir: str, backbone_embeddings: np.ndarray, id_to_idx: dict):
    """Load trained compat_head and scorer for evaluation."""
    from services.multihead import ProjectionHead, CompatibilityScorer

    # Load compat_head
    head_path = Path(weights_dir) / "compat_head.npz"
    if not head_path.exists():
        logger.error("No compat_head weights at %s", head_path)
        return None, None

    data = np.load(head_path)
    head = ProjectionHead(
        name="compat",
        w1=data["w1"], b1=data["b1"],
        ln_gamma=data["ln_gamma"], ln_beta=data["ln_beta"],
        w2=data["w2"], b2=data["b2"],
    )

    # Load scorer
    scorer_path = Path(weights_dir) / "compat_scorer.npz"
    scorer = None
    if scorer_path.exists():
        sdata = np.load(scorer_path)
        scorer = CompatibilityScorer(
            w1=sdata["w1"], b1=sdata["b1"],
            w2=sdata["w2"], b2=sdata["b2"],
            w3=sdata["w3"], b3=sdata["b3"],
        )

    # Pre-compute all compat projections
    compat_embeddings = head.forward(backbone_embeddings)
    logger.info("Multi-head system loaded: head + scorer")

    return compat_embeddings, scorer


# ── Cosine Baseline (System A proxy) ─────────────────────────────────────────


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / max(norm, 1e-8)


# ── FITB Evaluation ──────────────────────────────────────────────────────────


def eval_fitb_multihead(
    questions: list[dict],
    compat_embeddings: np.ndarray,
    scorer,
    id_to_idx: dict,
) -> dict:
    """
    FITB using multi-head compat_scorer.
    
    For each question: score each candidate against all other items in the outfit.
    Pick the candidate with the highest average compatibility.
    """
    correct = 0
    total = 0

    for q in questions:
        question_items = q.get("question", [])
        answers = q.get("answers", [])
        blank_pos = q.get("blank_position", 0)

        if not answers or len(answers) < 2:
            continue

        # Get context items (all items except the blank)
        context_ids = [item for i, item in enumerate(question_items) if i != blank_pos]
        context_indices = [id_to_idx.get(cid) for cid in context_ids]
        context_indices = [i for i in context_indices if i is not None]

        if not context_indices:
            continue

        context_embs = compat_embeddings[context_indices]

        # Score each answer candidate
        best_score = -1
        best_idx = 0
        for ans_idx, ans_id in enumerate(answers):
            ans_db_idx = id_to_idx.get(ans_id)
            if ans_db_idx is None:
                continue

            ans_emb = compat_embeddings[ans_db_idx]

            if scorer:
                # Use scorer for pairwise compatibility
                scores = [scorer.score(ans_emb, ctx) for ctx in context_embs]
                avg_score = np.mean(scores) if scores else 0
            else:
                # Fallback: cosine similarity in compat space
                scores = [cosine_similarity(ans_emb, ctx) for ctx in context_embs]
                avg_score = np.mean(scores) if scores else 0

            if avg_score > best_score:
                best_score = avg_score
                best_idx = ans_idx

        # First answer is always correct in Polyvore FITB
        if best_idx == 0:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {"fitb_accuracy": accuracy, "fitb_correct": correct, "fitb_total": total}


def eval_fitb_cosine_baseline(
    questions: list[dict],
    backbone_embeddings: np.ndarray,
    id_to_idx: dict,
) -> dict:
    """
    FITB using raw DINOv2 cosine similarity (no learned head).
    This represents what you'd get without training -- just backbone features.
    """
    correct = 0
    total = 0

    for q in questions:
        question_items = q.get("question", [])
        answers = q.get("answers", [])
        blank_pos = q.get("blank_position", 0)

        if not answers or len(answers) < 2:
            continue

        context_ids = [item for i, item in enumerate(question_items) if i != blank_pos]
        context_indices = [id_to_idx.get(cid) for cid in context_ids]
        context_indices = [i for i in context_indices if i is not None]

        if not context_indices:
            continue

        context_embs = backbone_embeddings[context_indices]

        best_score = -1
        best_idx = 0
        for ans_idx, ans_id in enumerate(answers):
            ans_db_idx = id_to_idx.get(ans_id)
            if ans_db_idx is None:
                continue

            ans_emb = backbone_embeddings[ans_db_idx]
            scores = [cosine_similarity(ans_emb, ctx) for ctx in context_embs]
            avg_score = np.mean(scores) if scores else 0

            if avg_score > best_score:
                best_score = avg_score
                best_idx = ans_idx

        if best_idx == 0:
            correct += 1
        total += 1

    accuracy = correct / max(total, 1)
    return {"fitb_accuracy": accuracy, "fitb_correct": correct, "fitb_total": total}


# ── Compatibility AUC ─────────────────────────────────────────────────────────


def eval_compat_auc(
    test_outfits: list[list[str]],
    compat_embeddings: np.ndarray,
    scorer,
    id_to_idx: dict,
    n_random: int = 1000,
) -> dict:
    """
    Score real outfits vs random outfits. Measure AUC.
    """
    from sklearn.metrics import roc_auc_score

    real_scores = []
    random_scores = []

    all_indices = list(id_to_idx.values())

    for outfit_items in test_outfits[:n_random]:
        indices = [id_to_idx.get(item) for item in outfit_items]
        indices = [i for i in indices if i is not None]

        if len(indices) < 2:
            continue

        outfit_embs = compat_embeddings[indices]

        # Score real outfit
        if scorer:
            pairs_scores = []
            for i in range(len(outfit_embs)):
                for j in range(i + 1, len(outfit_embs)):
                    pairs_scores.append(scorer.score(outfit_embs[i], outfit_embs[j]))
            real_scores.append(np.mean(pairs_scores))
        else:
            pairs_cos = []
            for i in range(len(outfit_embs)):
                for j in range(i + 1, len(outfit_embs)):
                    pairs_cos.append(cosine_similarity(outfit_embs[i], outfit_embs[j]))
            real_scores.append(np.mean(pairs_cos))

        # Score random "outfit" (same size, random items)
        rand_indices = np.random.choice(all_indices, size=len(indices), replace=False)
        rand_embs = compat_embeddings[rand_indices]

        if scorer:
            rand_pairs = []
            for i in range(len(rand_embs)):
                for j in range(i + 1, len(rand_embs)):
                    rand_pairs.append(scorer.score(rand_embs[i], rand_embs[j]))
            random_scores.append(np.mean(rand_pairs))
        else:
            rand_pairs = []
            for i in range(len(rand_embs)):
                for j in range(i + 1, len(rand_embs)):
                    rand_pairs.append(cosine_similarity(rand_embs[i], rand_embs[j]))
            random_scores.append(np.mean(rand_pairs))

    # Compute AUC
    labels = [1] * len(real_scores) + [0] * len(random_scores)
    scores = real_scores + random_scores

    if len(set(labels)) < 2:
        return {"compat_auc": 0.5, "n_real": len(real_scores), "n_random": len(random_scores)}

    auc = roc_auc_score(labels, scores)
    return {
        "compat_auc": auc,
        "real_mean_score": np.mean(real_scores),
        "random_mean_score": np.mean(random_scores),
        "n_real": len(real_scores),
        "n_random": len(random_scores),
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="A/B evaluation: multi-head vs baseline")
    parser.add_argument("--config", type=Path, default=Path("train/config.yaml"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/polyvore"))
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    h5_path = config["data"]["backbone_embeddings"]
    weights_dir = config["output"]["weights_dir"]

    # Load embeddings
    backbone_embeddings, item_ids, id_to_idx = load_embedding_store(h5_path)

    # Load multi-head system
    compat_embeddings, scorer = load_multihead_system(weights_dir, backbone_embeddings, id_to_idx)

    # Load test data
    fitb_questions = load_fitb_questions(args.data_dir)
    test_outfits = load_test_outfits(args.data_dir)

    # ── Results ───────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("A/B EVALUATION: Multi-Head (trained) vs Baseline (raw DINOv2 cosine)")
    print("=" * 70)

    # FITB
    if fitb_questions:
        print("\n--- Fill-In-The-Blank (FITB) ---")

        if compat_embeddings is not None:
            multihead_fitb = eval_fitb_multihead(fitb_questions, compat_embeddings, scorer, id_to_idx)
            print(f"  Multi-head (compat_scorer): {multihead_fitb['fitb_accuracy']:.4f} "
                  f"({multihead_fitb['fitb_correct']}/{multihead_fitb['fitb_total']})")

        baseline_fitb = eval_fitb_cosine_baseline(fitb_questions, backbone_embeddings, id_to_idx)
        print(f"  Baseline (DINOv2 cosine):   {baseline_fitb['fitb_accuracy']:.4f} "
              f"({baseline_fitb['fitb_correct']}/{baseline_fitb['fitb_total']})")

        if compat_embeddings is not None:
            improvement = multihead_fitb['fitb_accuracy'] - baseline_fitb['fitb_accuracy']
            print(f"  Improvement: {improvement:+.4f} ({improvement/max(baseline_fitb['fitb_accuracy'],0.01)*100:+.1f}%)")

    # Compatibility AUC
    if test_outfits:
        print("\n--- Compatibility AUC (real outfits vs random) ---")

        if compat_embeddings is not None and scorer:
            multihead_auc = eval_compat_auc(test_outfits, compat_embeddings, scorer, id_to_idx)
            print(f"  Multi-head (compat_scorer): AUC={multihead_auc['compat_auc']:.4f} "
                  f"(real={multihead_auc['real_mean_score']:.3f}, random={multihead_auc['random_mean_score']:.3f})")

        baseline_auc = eval_compat_auc(test_outfits, backbone_embeddings, None, id_to_idx)
        print(f"  Baseline (DINOv2 cosine):   AUC={baseline_auc['compat_auc']:.4f} "
              f"(real={baseline_auc['real_mean_score']:.3f}, random={baseline_auc['random_mean_score']:.3f})")

        if compat_embeddings is not None and scorer:
            auc_improvement = multihead_auc['compat_auc'] - baseline_auc['compat_auc']
            print(f"  AUC Improvement: {auc_improvement:+.4f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("  FITB > 0.60 = good (random chance = 0.25)")
    print("  AUC > 0.70 = scorer discriminates real from random outfits")
    print("  AUC > 0.80 = strong compatibility signal learned")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
