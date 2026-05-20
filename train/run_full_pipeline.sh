#!/usr/bin/env bash
# Full training pipeline: prepare data -> merge -> precompute -> train -> backfill -> eval
#
# Usage:
#   ./train/run_full_pipeline.sh                    # full pipeline
#   ./train/run_full_pipeline.sh --skip-scrape       # skip Pinterest scraping
#   ./train/run_full_pipeline.sh --from merge        # start from merge step
#   ./train/run_full_pipeline.sh --from train        # start from training
#   ./train/run_full_pipeline.sh --from backfill     # just re-backfill
#
set -euo pipefail

PYTHON="${PYTHON:-./venv/bin/python3}"
START_FROM="prepare"
SKIP_SCRAPE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-scrape)
            SKIP_SCRAPE="--skip-scrape"
            shift
            ;;
        --from)
            START_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "========================================"
echo "  Loom Multi-Head Training Pipeline"
echo "  Starting from: $START_FROM"
echo "========================================"

should_run() {
    local steps=("prepare" "merge" "precompute" "train" "backfill" "eval")
    local start_idx=-1
    local check_idx=-1
    for i in "${!steps[@]}"; do
        if [[ "${steps[$i]}" == "$START_FROM" ]]; then start_idx=$i; fi
        if [[ "${steps[$i]}" == "$1" ]]; then check_idx=$i; fi
    done
    [[ $check_idx -ge $start_idx ]]
}

# Step 1: Prepare data sources
if should_run "prepare"; then
    echo ""
    echo "=== Step 1a: Preparing Pinterest data ==="
    $PYTHON train/data/prepare_pinterest.py --output-dir data/pinterest --n-pins 5000 $SKIP_SCRAPE || echo "  Pinterest prep skipped (may need pinterest-crawler installed)"

    echo ""
    echo "=== Step 1b: Preparing FashionStylist data ==="
    $PYTHON train/data/prepare_fashionstylist.py --output-dir data/fashionstylist

    echo ""
    echo "=== Step 1c: Preparing FashionRec data ==="
    $PYTHON train/data/prepare_fashionrec.py --output-dir data/fashionrec

    echo ""
    echo "=== Step 1d: Preparing Polyvore data (if not already done) ==="
    if [ ! -f "data/polyvore/compat_pairs.csv" ]; then
        $PYTHON train/data/prepare_polyvore.py --data-dir data/polyvore --download
    else
        echo "  Polyvore data already exists, skipping."
    fi

    echo ""
    echo "=== Step 1e: Preparing DeepFashion data (if not already done) ==="
    if [ ! -f "data/deepfashion/occasion_pairs.csv" ]; then
        $PYTHON train/data/prepare_deepfashion.py --data-dir data/deepfashion --download || echo "  DeepFashion prep skipped"
    else
        echo "  DeepFashion data already exists, skipping."
    fi
fi

# Step 2: Merge all sources
if should_run "merge"; then
    echo ""
    echo "=== Step 2: Merging all data sources ==="
    $PYTHON train/data/merge_datasets.py --output-dir data/merged
fi

# Step 3: Precompute DINOv2 backbones
if should_run "precompute"; then
    echo ""
    echo "=== Step 3: Precomputing DINOv2 backbone embeddings ==="
    $PYTHON train/precompute_backbones.py --data-dir data/merged --output data/merged/backbone_embeddings.h5
fi

# Step 4: Train all heads
if should_run "train"; then
    echo ""
    echo "=== Step 4: Training all projection heads ==="
    $PYTHON train/train_heads.py --config train/config.yaml --head all
fi

# Step 5: Backfill production database
if should_run "backfill"; then
    echo ""
    echo "=== Step 5: Backfilling embeddings in production DB ==="
    $PYTHON scripts/backfill_multihead.py
fi

# Step 6: Run evaluation
if should_run "eval"; then
    echo ""
    echo "=== Step 6: Running human evaluation ==="
    $PYTHON eval/human_eval_export.py --n-samples 20
fi

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "  New weights: models/multihead/"
echo "  Human eval: eval/results/"
echo "========================================"
