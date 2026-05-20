"""
Contrastive training for multi-head projection layers.

Trains each head independently using InfoNCE loss on pre-computed DINOv2 embeddings.
The compat_head additionally trains a joint compatibility scorer (BCE loss).

Architecture per head:
    Linear(768, 256) → ReLU → LayerNorm → Linear(256, 128) → L2 Norm

Compatibility scorer (trained jointly with compat_head):
    [compat_a ; compat_b] (256) → Linear(256, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid

Usage:
    python train/train_heads.py --config train/config.yaml --head compat
    python train/train_heads.py --config train/config.yaml --head all
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Data Loading ─────────────────────────────────────────────────────────────


class EmbeddingStore:
    """Memory-mapped HDF5 store for pre-computed backbone embeddings."""

    def __init__(self, h5_path: str):
        import h5py
        self.f = h5py.File(h5_path, "r")
        self.embeddings = self.f["embeddings"]
        self.item_ids = [x.decode() if isinstance(x, bytes) else x for x in self.f["item_ids"][:]]
        self.id_to_idx = {item_id: i for i, item_id in enumerate(self.item_ids)}
        logger.info("Loaded embedding store: %d items, dim=%d", len(self.item_ids), self.embeddings.shape[1])

    def get(self, item_id: str) -> np.ndarray | None:
        idx = self.id_to_idx.get(item_id)
        if idx is None:
            return None
        return self.embeddings[idx]

    def get_batch(self, item_ids: list[str]) -> tuple[np.ndarray, list[int]]:
        """Get embeddings for a batch of IDs. Returns (embeddings, valid_indices)."""
        indices = []
        valid_positions = []
        for i, item_id in enumerate(item_ids):
            idx = self.id_to_idx.get(item_id)
            if idx is not None:
                indices.append(idx)
                valid_positions.append(i)

        if not indices:
            return np.zeros((0, self.embeddings.shape[1])), []

        # HDF5 requires sorted indices for efficient access
        sorted_order = np.argsort(indices)
        sorted_indices = [indices[j] for j in sorted_order]
        embeddings = self.embeddings[sorted_indices]

        # Unsort back to original order
        unsorted = np.empty_like(embeddings)
        for new_pos, orig_pos in enumerate(sorted_order):
            unsorted[orig_pos] = embeddings[new_pos]

        return unsorted, valid_positions

    def close(self):
        self.f.close()


class PairDataset:
    """Loads pair CSVs and yields batches of (emb_a, emb_b, label)."""

    def __init__(self, csv_path: str, store: EmbeddingStore, batch_size: int, shuffle: bool = True):
        self.store = store
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pairs = []
        self._positive_pairs = []
        self._negative_pairs = []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["item_a"] in store.id_to_idx and row["item_b"] in store.id_to_idx:
                    triple = (row["item_a"], row["item_b"], int(row["label"]))
                    self.pairs.append(triple)
                    if triple[2] == 1:
                        self._positive_pairs.append(triple)
                    else:
                        self._negative_pairs.append(triple)

        logger.info("PairDataset: %d valid pairs from %s", len(self.pairs), csv_path)

    def __len__(self):
        return len(self.pairs) // self.batch_size

    def iter_batches(self):
        """Yield (batch_emb_a, batch_emb_b, batch_labels) tuples."""
        indices = np.arange(len(self.pairs))
        if self.shuffle:
            np.random.shuffle(indices)

        for batch_start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
            batch_idx = indices[batch_start:batch_start + self.batch_size]
            batch_pairs = [self.pairs[i] for i in batch_idx]

            ids_a = [p[0] for p in batch_pairs]
            ids_b = [p[1] for p in batch_pairs]
            labels = np.array([p[2] for p in batch_pairs], dtype=np.float32)

            emb_a = np.array([self.store.get(id_) for id_ in ids_a])
            emb_b = np.array([self.store.get(id_) for id_ in ids_b])

            yield emb_a, emb_b, labels

    def replace_negatives_with_hard(self, hard_negatives: list[tuple]):
        """Swap random negatives for hard-mined ones (partial replacement)."""
        n_replace = min(len(hard_negatives), len(self._negative_pairs) // 2)
        if n_replace == 0:
            return

        kept_negatives = self._negative_pairs[n_replace:]
        self._negative_pairs = kept_negatives + hard_negatives[:n_replace]
        self.pairs = self._positive_pairs + self._negative_pairs
        np.random.shuffle(self.pairs)
        logger.info("  Replaced %d negatives with hard-mined samples", n_replace)


# ── Model Definitions (Torch) ────────────────────────────────────────────────


def _build_head_model():
    """Build a trainable projection head in PyTorch."""
    import torch
    import torch.nn as nn

    class ProjectionHead(nn.Module):
        def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.relu = nn.ReLU()
            self.ln = nn.LayerNorm(hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            h = self.fc1(x)
            h = self.relu(h)
            h = self.ln(h)
            out = self.fc2(h)
            out = torch.nn.functional.normalize(out, p=2, dim=-1)
            return out

    return ProjectionHead()


def _build_scorer_model(input_dim=256, hidden_dims=None):
    """Build compatibility scorer MLP."""
    import torch
    import torch.nn as nn

    if hidden_dims is None:
        hidden_dims = [128, 64]

    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, dim))
        layers.append(nn.ReLU())
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, 1))
    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)


# ── Loss Functions ────────────────────────────────────────────────────────────


def info_nce_loss(anchor, positive, temperature=0.07):
    """
    InfoNCE contrastive loss.
    
    Within a batch of N pairs, each anchor has 1 positive and (N-1) negatives
    (all other positives in the batch serve as in-batch negatives).
    """
    import torch
    import torch.nn.functional as F

    # Similarity matrix: (N, N)
    anchor_norm = F.normalize(anchor, p=2, dim=-1)
    positive_norm = F.normalize(positive, p=2, dim=-1)

    # Positive similarities (diagonal)
    pos_sim = (anchor_norm * positive_norm).sum(dim=-1) / temperature

    # All similarities (each anchor vs all positives as potential matches)
    all_sim = torch.mm(anchor_norm, positive_norm.T) / temperature

    # InfoNCE: log(exp(pos) / sum(exp(all)))
    # = pos_sim - logsumexp(all_sim)
    loss = -pos_sim + torch.logsumexp(all_sim, dim=-1)
    return loss.mean()


def bce_loss_fn(predictions, labels):
    """Binary cross-entropy for compatibility scorer."""
    import torch.nn.functional as F
    return F.binary_cross_entropy(predictions.squeeze(), labels)


# ── Hard Negative Mining ──────────────────────────────────────────────────────


def mine_hard_negatives(
    head,
    dataset: PairDataset,
    n_hard: int = 5000,
    device: str = "cpu",
) -> list[tuple]:
    """
    Mine hard negatives: negative pairs that the current model scores highly.

    For each positive anchor, find the highest-scoring negative in a random pool.
    These are the pairs the model is most confused about, so training on them
    gives the strongest gradient signal.
    """
    import torch

    head.eval()
    store = dataset.store
    all_ids = store.item_ids
    positives = dataset._positive_pairs

    if not positives or not all_ids:
        return []

    pool_size = min(500, len(all_ids))
    hard_negs = []

    sample_anchors = positives[:min(n_hard * 2, len(positives))]
    np.random.shuffle(sample_anchors)

    with torch.no_grad():
        for anchor_id, _, _ in sample_anchors[:n_hard]:
            anchor_emb = store.get(anchor_id)
            if anchor_emb is None:
                continue

            candidates = np.random.choice(len(all_ids), size=pool_size, replace=False)
            cand_ids = [all_ids[c] for c in candidates]
            cand_embs = np.array([store.get(cid) for cid in cand_ids if store.get(cid) is not None])

            if len(cand_embs) < 2:
                continue

            anchor_t = torch.tensor(anchor_emb, dtype=torch.float32, device=device).unsqueeze(0)
            cand_t = torch.tensor(cand_embs, dtype=torch.float32, device=device)

            proj_anchor = head(anchor_t)
            proj_cand = head(cand_t)

            sims = torch.mm(proj_anchor, proj_cand.T).squeeze(0)
            hardest_idx = sims.argmax().item()
            hard_negs.append((anchor_id, cand_ids[hardest_idx], 0))

            if len(hard_negs) >= n_hard:
                break

    logger.info("  Mined %d hard negatives", len(hard_negs))
    head.train()
    return hard_negs


# ── Training Loop ─────────────────────────────────────────────────────────────


def train_single_head(
    head_name: str,
    dataset: PairDataset,
    config: dict,
    device: str = "cpu",
) -> tuple:
    """
    Train one projection head with InfoNCE loss.
    If head_name == "compat", also trains the compatibility scorer jointly.
    
    Returns (trained_head, trained_scorer_or_None).
    """
    import torch
    import torch.optim as optim

    head_overrides = config.get("head_overrides", {}).get(head_name, {})
    epochs = head_overrides.get("epochs", config["training"]["epochs"])
    lr = config["training"]["learning_rate"]
    temperature = head_overrides.get("temperature", config["training"]["temperature"])
    warmup_epochs = config["training"]["warmup_epochs"]
    patience = config["training"]["patience"]
    scorer_weight = config["training"]["scorer_weight"]
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build models
    head = _build_head_model().to(device)
    scorer = None
    if head_name == "compat":
        scorer_hidden = config["model"]["scorer_hidden_dims"]
        scorer = _build_scorer_model(
            input_dim=config["model"]["output_dim"] * 2,
            hidden_dims=scorer_hidden,
        ).to(device)

    # Optimizer
    params = list(head.parameters())
    if scorer is not None:
        params += list(scorer.parameters())
    optimizer = optim.AdamW(params, lr=lr, weight_decay=config["training"]["weight_decay"])

    # Hard negative mining config
    use_hard_neg = config["training"].get("hard_negative_mining", False)
    hard_neg_start = config["training"].get("hard_neg_start_epoch", 10)

    # Training
    best_val_loss = float("inf")
    patience_counter = 0

    logger.info("Training %s_head: %d epochs, lr=%.1e, tau=%.3f, device=%s, hard_neg=%s (epoch %d+)",
                head_name, epochs, lr, temperature, device, use_hard_neg, hard_neg_start)

    for epoch in range(epochs):
        # Hard negative mining: replace easy negatives with hard ones
        if use_hard_neg and epoch == hard_neg_start:
            logger.info("Epoch %d: mining hard negatives...", epoch + 1)
            hard_negs = mine_hard_negatives(head, dataset, n_hard=5000, device=device)
            if hard_negs:
                dataset.replace_negatives_with_hard(hard_negs)

        head.train()
        if scorer:
            scorer.train()

        epoch_loss = 0.0
        epoch_contrastive = 0.0
        epoch_scorer = 0.0
        n_batches = 0

        # Linear warmup
        if epoch < warmup_epochs:
            lr_scale = (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr * lr_scale

        for emb_a, emb_b, labels in dataset.iter_batches():
            emb_a_t = torch.tensor(emb_a, dtype=torch.float32, device=device)
            emb_b_t = torch.tensor(emb_b, dtype=torch.float32, device=device)
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            # Project through head
            proj_a = head(emb_a_t)
            proj_b = head(emb_b_t)

            # Contrastive loss (only on positive pairs for InfoNCE)
            pos_mask = labels_t == 1
            if pos_mask.sum() < 2:
                continue

            contrastive = info_nce_loss(
                proj_a[pos_mask], proj_b[pos_mask], temperature
            )

            total_loss = contrastive
            epoch_contrastive += contrastive.item()

            # Scorer loss (for compat_head)
            if scorer is not None:
                scorer_input = torch.cat([proj_a, proj_b], dim=-1)
                predictions = scorer(scorer_input)
                s_loss = bce_loss_fn(predictions, labels_t)
                total_loss = total_loss + scorer_weight * s_loss
                epoch_scorer += s_loss.item()

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += total_loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_contrastive = epoch_contrastive / max(n_batches, 1)

        log_msg = f"Epoch {epoch+1}/{epochs} | loss={avg_loss:.4f} | contrastive={avg_contrastive:.4f}"
        if scorer:
            avg_scorer = epoch_scorer / max(n_batches, 1)
            log_msg += f" | scorer_bce={avg_scorer:.4f}"
        logger.info(log_msg)

        # Simple early stopping on training loss (use val split for real training)
        if avg_loss < best_val_loss - 1e-4:
            best_val_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Checkpoint
        save_every = config["training"].get("save_every_epochs", 5)
        if (epoch + 1) % save_every == 0:
            ckpt_path = checkpoint_dir / f"{head_name}_head_epoch{epoch+1}.pt"
            torch.save(head.state_dict(), ckpt_path)
            logger.info("Checkpoint saved: %s", ckpt_path)

    return head, scorer


def export_head_to_npz(head, head_name: str, output_dir: Path):
    """Convert trained PyTorch head to numpy .npz for production inference."""
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    state = head.state_dict()
    np.savez(
        output_dir / f"{head_name}_head.npz",
        w1=state["fc1.weight"].T.cpu().numpy(),        # Transpose: PyTorch stores (out, in)
        b1=state["fc1.bias"].cpu().numpy(),
        ln_gamma=state["ln.weight"].cpu().numpy(),
        ln_beta=state["ln.bias"].cpu().numpy(),
        w2=state["fc2.weight"].T.cpu().numpy(),
        b2=state["fc2.bias"].cpu().numpy(),
    )
    logger.info("Exported %s_head to %s", head_name, output_dir / f"{head_name}_head.npz")


def export_scorer_to_npz(scorer, output_dir: Path):
    """Convert trained scorer to numpy .npz for production inference."""
    output_dir.mkdir(parents=True, exist_ok=True)

    state = scorer.state_dict()
    # Sequential layers: 0=Linear, 1=ReLU, 2=Linear, 3=ReLU, 4=Linear, 5=Sigmoid
    np.savez(
        output_dir / "compat_scorer.npz",
        w1=state["0.weight"].T.cpu().numpy(),
        b1=state["0.bias"].cpu().numpy(),
        w2=state["2.weight"].T.cpu().numpy(),
        b2=state["2.bias"].cpu().numpy(),
        w3=state["4.weight"].T.cpu().numpy(),
        b3=state["4.bias"].cpu().numpy(),
    )
    logger.info("Exported compat_scorer to %s", output_dir / "compat_scorer.npz")


# ── Main ──────────────────────────────────────────────────────────────────────


HEAD_TO_PAIRS = {
    "compat": "compat_pairs",
    "style": "style_pairs",
    "occasion": "occasion_pairs",
    "fit": "fit_pairs",
    "material": "material_pairs",
}


def main():
    import torch

    parser = argparse.ArgumentParser(description="Train multi-head projection layers")
    parser.add_argument("--config", type=Path, default=Path("train/config.yaml"))
    parser.add_argument("--head", type=str, default="all",
                       choices=["all", "compat", "style", "occasion", "fit", "material"])
    parser.add_argument("--device", type=str, default=None,
                       help="Device (cpu/cuda/mps). Auto-detected if not set.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Auto-detect device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # Load pre-computed embeddings
    store = EmbeddingStore(config["data"]["backbone_embeddings"])

    heads_to_train = list(HEAD_TO_PAIRS.keys()) if args.head == "all" else [args.head]
    output_dir = Path(config["output"]["weights_dir"])

    for head_name in heads_to_train:
        pairs_key = HEAD_TO_PAIRS[head_name]
        pairs_path = config["data"].get(pairs_key)

        if not pairs_path or not Path(pairs_path).exists():
            logger.warning("Pairs file not found for %s_head (%s), skipping", head_name, pairs_path)
            continue

        head_overrides = config.get("head_overrides", {}).get(head_name, {})
        batch_size = head_overrides.get("batch_size", config["training"]["batch_size"])

        dataset = PairDataset(pairs_path, store, batch_size=batch_size)

        if len(dataset) == 0:
            logger.warning("No valid pairs for %s_head, skipping", head_name)
            continue

        t0 = time.time()
        trained_head, trained_scorer = train_single_head(head_name, dataset, config, device)
        elapsed = time.time() - t0
        logger.info("%s_head training complete in %.1f minutes", head_name, elapsed / 60)

        # Export to production format
        export_head_to_npz(trained_head, head_name, output_dir)

        if trained_scorer is not None:
            export_scorer_to_npz(trained_scorer, output_dir)

    store.close()
    logger.info("All done. Weights saved to %s", output_dir)


if __name__ == "__main__":
    main()
