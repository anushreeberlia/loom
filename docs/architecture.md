# Loom Architecture: Current State & Northstar

## Overview

Loom is an AI outfit styling system that generates complete outfits from a user's closet or catalog items. This document describes the current production architecture and the northstar vision for a learned, multi-head representation system.

---

## Current Architecture (Production)

### Pipeline Summary

```
User uploads photo/video
    → U2-Net segmentation (background removal)
    → Fashion Florence (structured metadata extraction)
    → FashionCLIP 2.0 (single 512-dim blended embedding)
    → pgvector storage + HNSW index
    → Retrieval: text query → vector search → manual re-ranking
    → Outfit assembly: 4-level hierarchical scoring
    → 3 outfits (Classic / Trendy / Bold)
```

### Components

#### 1. Item Ingestion

| Step | Tool | Output |
|------|------|--------|
| Object detection | YOLO + ByteTrack | Per-object crops from video/camera |
| Segmentation | U2-Net (rembg, isnet-general-use) | Clean garment cutout |
| Metadata extraction | Fashion Florence (HF Space) | category, color, material, style_tags, fit, occasion_tags, season_tags |
| Embedding | FashionCLIP 2.0 (ONNX) | 512-dim vector (0.7 * image + 0.3 * text metadata) |
| Temporal aggregation | Quality-weighted avg of top-5 crops | Single robust embedding from multiple video frames |

#### 2. Retrieval (`services/retrieval.py` -- 1196 lines)

- **Vector search**: `embedding <-> query::vector` via pgvector HNSW (cosine ops)
- **Occasion scoring**: Embed "vibe" and "anti-vibe" text descriptions, compute cosine similarity to item embedding, apply hardcoded penalties per occasion
- **Direction re-ranking**: Rule-based color/tag/keyword scoring for Classic/Trendy/Bold
- **Color filtering**: Hard avoid + soft preference + neutral boosting + 5% noise
- **Tag-mood scoring**: Per-tag cosine similarity to free-form mood text

#### 3. Outfit Scoring (`services/outfit.py` -- 2070 lines)

Hierarchical 4-level scoring (30% / 25% / 25% / 20% weighted):

| Level | What it checks | Mechanism |
|-------|---------------|-----------|
| L1: Silhouette | Proportion balance, shoe-bottom harmony, embedding spread | Rule-based proportion checks |
| L2: Color Surfaces | Color composition, bookend score, calm, adjacent clash, embedding harmony | Hardcoded color theory rules |
| L3: Texture + Narrative | Texture contrast/variety, hero clarity, gradient, deference | Material keyword matching |
| L4: Finishing | Intent alignment, direction bonus, formality coherence, occasion coherence | Cosine to intent vector + rules |

Gating: weak L1 dampens L2-L4 by 0.5x; weak L2 dampens L3-L4 by 0.7x.

#### 4. Personalization (`taste_vectors` table)

- Moving average of liked outfit embeddings (512-dim)
- Separate dislike embedding (512-dim)
- Blended into intent vector: `0.6*base + 0.15*taste - 0.05*dislike`
- Same embedding space as items (no task-specific projection)

### Limitations

1. **Single embedding for all tasks**: One 512-dim vector answers "visually similar?", "work appropriate?", "goes with jeans?", and "user's style?" -- it can't separate these axes.

2. **FashionCLIP's training objective**: Trained on image-caption matching ("blue casual cotton top"). Does NOT encode compatibility, occasion appropriateness, or styling relationships.

3. **Manual scoring compensation**: 1196 lines of retrieval hacks + 2070 lines of outfit scoring exist because the embedding can't differentiate a crop top from a blazer for work-appropriateness (both are "women's upper body garment," cosine ~0.7).

4. **No learned compatibility**: "Does this top go with these pants?" is answered by color rules and texture keyword matching, not by learning from actual outfit data.

5. **Crude preference model**: Moving average in the item embedding space. No task-specific preference learning.

---

## Northstar Architecture

### Core Idea

Replace the single shared embedding with a **frozen visual backbone** (DINOv2) + **task-specific MLP projection heads**, each trained with contrastive learning to separate different notions of similarity.

### Pipeline Summary

```
Segmented image
    → DINOv2 ViT-B/14 (frozen backbone, 768-dim)
    → Shared embedding E
    → MLP projection heads (each 768 → 256 → 128):
        style_vec   = style_head(E)      -- minimalist vs streetwear vs romantic
        fit_vec     = fit_head(E)        -- oversized vs tailored vs bodycon
        material_vec = material_head(E)  -- wool vs denim vs silk vs knit
        compat_vec  = compat_head(E)     -- "goes well with" representation
        occasion_vec = occasion_head(E)  -- work vs casual vs going-out

    → Compatibility Scorer MLP:
        input: [compat_A ; compat_B] (256-dim concatenation)
        output: compatibility score (0-1)

    → User Preference Head:
        trained on liked/disliked items
        produces user_pref_vec (128-dim)
```

### Why DINOv2

| Property | FashionCLIP | DINOv2 |
|----------|-------------|--------|
| Training | Image-caption matching (product descriptions) | Self-supervised on 142M images (pixel-level structure) |
| Dimensions | 512 | 768 |
| Strength | Text-image alignment | Visual structure: texture, silhouette, pattern, spatial layout |
| Weakness | Can't distinguish styling relationships from visual similarity | No text understanding |
| For compatibility | Never saw co-worn pairs | Rich visual features are the right substrate for learning "these go together" |

DINOv2 captures texture differences (silk vs cotton, knit vs woven) that FashionCLIP ignores because those differences don't appear in product captions.

### Multi-Head Projection Heads

Each head is a small MLP (2-3 linear layers) that learns a **different notion of similarity** from the same backbone representation.

```
head architecture:
    Linear(768, 256) → ReLU → LayerNorm
    Linear(256, 128) → L2 normalize
```

**What each head learns:**

| Head | Separates | Example |
|------|-----------|---------|
| style_head | Aesthetic identity | minimalist vs streetwear (two jackets that look similar but have different style identities) |
| fit_head | Silhouette/cut | oversized hoodie vs tailored blazer (same "upper body layer" but completely different fits) |
| material_head | Fabric/texture | wool coat vs denim jacket (similar shape, completely different materials) |
| compat_head | Co-outfit suitability | Items that work together live close; clashing items are far apart |
| occasion_head | Context appropriateness | Crop top far from blazer in "work" space, close in "going-out" space |

### Training: Contrastive Learning

Each head is trained independently with **positive/negative pairs**, exactly like CLIP:

```
Loss: d(anchor, positive) < d(anchor, negative) - margin
```

This reshapes the latent geometry so that items similar along ONE axis (e.g., style) are close in that head's space, even if they're different along other axes.

#### Training Data Sources

| Source | Signal | Heads it trains |
|--------|--------|-----------------|
| **Polyvore outfits** (68K outfits, 365K items, public) | Co-outfit items = positive compatibility pairs | compat_head, style_head |
| **Fashion Florence tags** (weak supervision) | Same occasion/fit/material labels = positive | occasion_head, fit_head, material_head |
| **Pinterest aesthetic boards** | Same board = positive style pair | style_head |
| **"Complete the look" data** (ecommerce) | Recommended pairings = positive | compat_head |
| **feedback_events + saved_outfits** (your data) | Liked items = positive preference signal | preference_head |

Key insight: you do NOT need massive manual labeling. Existing outfit datasets implicitly encode compatibility + aesthetics.

#### Contrastive Training Example (style_head)

```python
# Positive pair: two items from the same Pinterest aesthetic board
anchor = style_head(dinov2(item_a))    # minimalist linen shirt
positive = style_head(dinov2(item_b))  # minimalist wide-leg trousers

# Negative: item from a different aesthetic board
negative = style_head(dinov2(item_c))  # streetwear graphic hoodie

# Triplet loss pushes: d(anchor, positive) < d(anchor, negative)
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

### Compatibility Scorer

A simple feedforward MLP that takes two items' compat_vecs and predicts how well they work together:

```python
class CompatibilityScorer:
    """
    Input: [item_a_compat_vec ; item_b_compat_vec] (256-dim)
    Output: scalar compatibility score (0-1)
    
    Architecture: Linear(256, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 1) → Sigmoid
    
    Training:
        Positive: (top, bottom) from same Polyvore outfit → label 1
        Negative: (top, random_bottom) → label 0
    """
```

This replaces the entire 2070-line `score_outfit` hierarchy with a learned signal. At outfit generation time:

```python
outfit_score = sum(
    compat_scorer(item_i, item_j)
    for i, j in all_pairs(outfit_items)
) + occasion_bonus + style_bonus + preference_bonus
```

### User Preference Layer

Trained on your `feedback_events` and `saved_outfits` data:

- Items in **liked** outfits → positive examples
- Items in **disliked** outfits → negative examples
- Learn a `preference_head` that projects items into a space where the user's preferred items cluster

Replaces the crude moving-average taste_vector with a learned projection that captures nuanced preference (not just "close to liked items in the same space where everything else lives").

### What Gets Eliminated

| Current (rule-based) | Northstar (learned) | Lines removed |
|---------------------|---------------------|---------------|
| `OCCASION_SEMANTIC_CONTEXTS` + vibe/anti-vibe | occasion_head cosine | ~100 |
| `compute_occasion_score` + anti-vibe penalties | occasion_vec dot product (1 line) | ~60 |
| `apply_direction_rerank` (Classic/Bold/Trendy rules) | style_vec similarity | ~120 |
| `check_proportion_balance` + `check_shoe_bottom_harmony` | fit_head learned silhouette compatibility | ~80 |
| `check_texture_contrast` + material keyword matching | material_head similarity | ~60 |
| `score_outfit` 4-level hierarchy | compat_scorer pairwise + head bonuses | ~300 |
| `filter_by_occasion_semantic` | pgvector query on occasion_embedding | ~100 |
| taste_vector moving average | Learned preference_head | ~50 |
| **Total** | | **~870 lines of manual scoring** |

### Storage Schema (Northstar)

```sql
-- Per item: 5 head embeddings (128-dim each) = 640 total dims
ALTER TABLE user_closet_items ADD COLUMN style_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN fit_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN material_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN compat_embedding vector(128);
ALTER TABLE user_closet_items ADD COLUMN occasion_embedding vector(128);

-- HNSW indexes on heads used for primary retrieval
CREATE INDEX idx_closet_compat USING hnsw (compat_embedding vector_cosine_ops);
CREATE INDEX idx_closet_style USING hnsw (style_embedding vector_cosine_ops);
CREATE INDEX idx_closet_occasion USING hnsw (occasion_embedding vector_cosine_ops);
```

### Migration Path

| Phase | What | Breaking changes? |
|-------|------|-------------------|
| 1 | Add DINOv2 encoder alongside FashionCLIP | None -- both run, old column stays |
| 2 | Add multi-head columns (nullable) + random-init heads | None -- retrieval uses old column if new is NULL |
| 3 | Train heads on Polyvore + Florence weak supervision | None -- training is offline |
| 4 | Deploy trained heads, update retrieval to use head-specific queries | Gradual -- A/B test against old scoring |
| 5 | Train compatibility scorer on Polyvore pairs | None -- new feature |
| 6 | Replace score_outfit with scorer-based ranking | Major -- remove 870 lines of rules |
| 7 | Train preference head on feedback_events | None -- new feature |
| 8 | Drop FashionCLIP image encoder + old embedding column | Cleanup |

### Key Design Decisions

- **Backbone is frozen**: DINOv2 never gets fine-tuned. Only the projection heads train. This means you can swap backbones later without retraining everything.
- **Heads are independent**: Each head trains on its own data/loss. No multi-task interference.
- **Contrastive over classification**: Triplet/InfoNCE loss creates continuous similarity spaces. Classification would require fixed categories.
- **Start simple, improve with data**: Random-init heads still pass through useful compressed features. Train to improve, not to function.
- **Polyvore first**: 68K outfits is enough to train a strong compatibility scorer. Your own feedback_events supplements but isn't the primary source.

---

## System Diagram (Full Northstar)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ITEM INGESTION                                      │
│                                                                                 │
│  Camera/Video/Photo                                                             │
│       │                                                                         │
│       ▼                                                                         │
│  YOLO + ByteTrack (multi-object tracking, per-object temporal buffer)           │
│       │                                                                         │
│       ▼                                                                         │
│  U2-Net Segmentation (clean garment cutout)                                     │
│       │                                                                         │
│       ├──────────────────────────┐                                              │
│       ▼                          ▼                                              │
│  DINOv2 ViT-B/14 (768-dim)     Fashion Florence (tags)                         │
│       │                          │                                              │
│       ▼                          │                                              │
│  ┌─────────────────────────┐     │                                              │
│  │ MULTI-HEAD PROJECTIONS  │     │                                              │
│  │                         │     │                                              │
│  │ style_head(E) → 128    │     │                                              │
│  │ fit_head(E) → 128      │     │                                              │
│  │ material_head(E) → 128 │     │                                              │
│  │ compat_head(E) → 128   │     │                                              │
│  │ occasion_head(E) → 128 │◄────┘  (occasion uses tag features too)            │
│  └─────────────────────────┘                                                    │
│       │                                                                         │
│       ▼                                                                         │
│  pgvector: 5 × vector(128) + HNSW indexes                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              OUTFIT GENERATION                                   │
│                                                                                 │
│  Input: base item + occasion + style direction + user preferences               │
│                                                                                 │
│  Step 1: RETRIEVAL (per slot)                                                   │
│     Query: base_item.compat_vec → pgvector search on compat_embedding           │
│     Filter: occasion_vec cosine > threshold                                     │
│     Boost: style_vec similarity to direction anchors                            │
│                                                                                 │
│  Step 2: COMPATIBILITY SCORING                                                  │
│     For each candidate outfit (combination of slot items):                      │
│       score = Σ compat_scorer(item_i, item_j) for all pairs                    │
│                                                                                 │
│  Step 3: OCCASION + STYLE BONUS                                                 │
│     + occasion_head similarity to occasion anchor                               │
│     + style_head similarity to direction anchor                                 │
│     + preference_head similarity to user_pref_vec                               │
│                                                                                 │
│  Step 4: SELECT BEST                                                            │
│     outfit_score = compat_total + occasion_bonus + style_bonus + pref_bonus     │
│     Return top outfit per direction                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING (Offline)                                   │
│                                                                                 │
│  Contrastive Head Training:                                                     │
│    Data: Polyvore (68K outfits) + Florence tags + Pinterest boards              │
│    Loss: Triplet / InfoNCE                                                      │
│    d(anchor, positive) < d(anchor, negative) - margin                           │
│                                                                                 │
│  Compatibility Scorer Training:                                                 │
│    Data: Polyvore positive pairs + random negatives                             │
│    Loss: Binary cross-entropy                                                   │
│    Input: [compat_vec_A ; compat_vec_B] → score 0-1                            │
│                                                                                 │
│  Preference Head Training:                                                      │
│    Data: feedback_events (liked/disliked items from outfits)                    │
│    Loss: Contrastive (liked items close, disliked items far)                    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Map

| File | Current Role | Northstar Role |
|------|-------------|----------------|
| `services/fashion_clip.py` | Image + text encoder (512-dim) | Text encoder only (for occasion queries) → eventually removed |
| `services/dinov2.py` | Does not exist | Frozen backbone encoder (768-dim) |
| `services/multihead.py` | Does not exist | 5 MLP projection heads + compatibility scorer |
| `services/embedding.py` | Blended embedding (0.7 img + 0.3 txt) | Routes to appropriate head for task |
| `services/retrieval.py` | 1196 lines of manual scoring | Head-routed pgvector queries (~200 lines) |
| `services/outfit.py` | 2070 lines of rule-based scoring | Compatibility scorer + head bonuses (~300 lines) |
| `services/item_processor.py` | Vision + single embedding | Vision + DINOv2 + all heads |
| `services/segmentation.py` | U2-Net background removal | Same (feeds cleaner input to DINOv2) |
| `services/object_tracker.py` | YOLO + ByteTrack + temporal aggregation | Same (upstream of embedding) |
| `train/train_heads.py` | Does not exist | Contrastive training loop for all heads |
| `train/train_compat.py` | Does not exist | Compatibility scorer training |
| `train/train_preference.py` | Does not exist | User preference head training |
