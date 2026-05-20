"""
Object tracking pipeline for video and live auto-capture.

Uses YOLO for per-frame detection and ByteTrack for multi-object tracking
across frames. Produces one best crop per unique tracked object.

Two modes:
- Video (batch): process_video_frames() runs all frames, returns best crops
- Live (streaming): ByteTracker.update() called per-frame, TrackedObject
  reports when stable enough to capture
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
import io

logger = logging.getLogger(__name__)

# ─── Quality scoring ───────────────────────────────────────────────────────────

def frame_sharpness(image_bytes: bytes, size: int = 256) -> float:
    """Laplacian variance as sharpness proxy (higher = sharper)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((size, size))
    laplacian = img.filter(ImageFilter.Kernel(
        size=(3, 3), kernel=[-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=128
    ))
    return float(np.var(np.array(laplacian, dtype=np.float32)))


def bbox_centeredness(bbox: tuple, frame_w: int, frame_h: int) -> float:
    """Score 0-1 based on how centered the bbox is in the frame."""
    x, y, w, h = bbox
    cx = (x + w / 2) / frame_w
    cy = (y + h / 2) / frame_h
    dist = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5
    return max(0.0, 1.0 - dist * 2)


def bbox_area_ratio(bbox: tuple, frame_w: int, frame_h: int) -> float:
    """Fraction of frame covered by bbox. Prefer larger objects."""
    _, _, w, h = bbox
    return (w * h) / (frame_w * frame_h)


def compute_quality_score(
    image_bytes: bytes, bbox: tuple, frame_w: int, frame_h: int
) -> float:
    """
    Combined quality score for a detection crop.
    Weights: sharpness 0.5, centeredness 0.3, size 0.2
    """
    sharpness = min(frame_sharpness(image_bytes) / 1000.0, 1.0)
    center = bbox_centeredness(bbox, frame_w, frame_h)
    area = min(bbox_area_ratio(bbox, frame_w, frame_h) * 5, 1.0)
    return sharpness * 0.5 + center * 0.3 + area * 0.2


# ─── ByteTrack ─────────────────────────────────────────────────────────────────

def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """IoU between two (x, y, w, h) bounding boxes."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix = max(0, min(ax2, bx2) - max(ax, bx))
    iy = max(0, min(ay2, by2) - max(ay, by))
    intersection = ix * iy

    union = aw * ah + bw * bh - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_matrix(tracks: list, detections: list) -> np.ndarray:
    """Cost matrix (1 - IoU) between tracks and detections."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    matrix = np.ones((n_tracks, n_dets), dtype=np.float32)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            matrix[i, j] = 1.0 - compute_iou(track.bbox, det["bbox"])

    return matrix


def greedy_match(cost_matrix: np.ndarray, threshold: float):
    """
    Greedy matching for real-time performance.
    Returns: matches (list of (track_idx, det_idx)), unmatched_tracks, unmatched_dets
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    n_tracks, n_dets = cost_matrix.shape
    matches = []
    used_tracks = set()
    used_dets = set()

    flat_indices = np.argsort(cost_matrix, axis=None)
    for flat_idx in flat_indices:
        i = flat_idx // n_dets
        j = flat_idx % n_dets
        if i in used_tracks or j in used_dets:
            continue
        if cost_matrix[i, j] > threshold:
            break
        matches.append((i, j))
        used_tracks.add(i)
        used_dets.add(j)

    unmatched_tracks = [i for i in range(n_tracks) if i not in used_tracks]
    unmatched_dets = [j for j in range(n_dets) if j not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


def hungarian_match(cost_matrix: np.ndarray, threshold: float):
    """
    Optimal matching using scipy's linear_sum_assignment.
    Falls back to greedy if scipy unavailable.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return greedy_match(cost_matrix, threshold)

    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    used_tracks = set()
    used_dets = set()

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] <= threshold:
            matches.append((r, c))
            used_tracks.add(r)
            used_dets.add(c)

    unmatched_tracks = [i for i in range(cost_matrix.shape[0]) if i not in used_tracks]
    unmatched_dets = [j for j in range(cost_matrix.shape[1]) if j not in used_dets]
    return matches, unmatched_tracks, unmatched_dets


MAX_TEMPORAL_CROPS = 5  # Top-K crops to keep for temporal aggregation


@dataclass
class Track:
    """A single tracked object across frames."""
    track_id: int
    bbox: tuple  # (x, y, w, h)
    confidence: float
    frames_seen: int = 1
    frames_lost: int = 0
    stable_count: int = 0
    last_center: tuple = (0.0, 0.0)

    # Best frame tracking
    best_crop_bytes: bytes = b""
    best_quality: float = 0.0
    captured: bool = False

    # Temporal buffer: top-K crops for embedding aggregation
    temporal_crops: list = field(default_factory=list)  # [(crop_bytes, quality_score)]

    # Stability
    _center_threshold: float = 0.05

    def update(self, detection: dict, frame_bytes: bytes = None, frame_w: int = 0, frame_h: int = 0):
        """Update track with new detection."""
        self.bbox = detection["bbox"]
        self.confidence = detection["confidence"]
        self.frames_seen += 1
        self.frames_lost = 0

        x, y, w, h = self.bbox
        center = ((x + w / 2) / max(frame_w, 1), (y + h / 2) / max(frame_h, 1))

        if self.last_center != (0.0, 0.0):
            dx = abs(center[0] - self.last_center[0])
            dy = abs(center[1] - self.last_center[1])
            if dx < self._center_threshold and dy < self._center_threshold:
                self.stable_count += 1
            else:
                self.stable_count = 0
        self.last_center = center

        if frame_bytes and frame_w > 0:
            self._update_best_frame(frame_bytes, frame_w, frame_h)

    def _update_best_frame(self, frame_bytes: bytes, frame_w: int, frame_h: int):
        """Keep top-K crops sorted by quality for temporal aggregation."""
        try:
            crop_bytes = self._crop_from_frame(frame_bytes, frame_w, frame_h)
            quality = compute_quality_score(crop_bytes, self.bbox, frame_w, frame_h)

            if quality > self.best_quality:
                self.best_quality = quality
                self.best_crop_bytes = crop_bytes

            # Maintain temporal buffer of top-K crops
            if len(self.temporal_crops) < MAX_TEMPORAL_CROPS:
                self.temporal_crops.append((crop_bytes, quality))
                self.temporal_crops.sort(key=lambda x: x[1], reverse=True)
            elif quality > self.temporal_crops[-1][1]:
                self.temporal_crops[-1] = (crop_bytes, quality)
                self.temporal_crops.sort(key=lambda x: x[1], reverse=True)
        except Exception:
            pass

    def _crop_from_frame(self, frame_bytes: bytes, frame_w: int, frame_h: int) -> bytes:
        """Crop bbox region from frame with padding."""
        img = Image.open(io.BytesIO(frame_bytes))
        x, y, w, h = self.bbox
        pad = max(w, h) * 0.12
        left = max(0, int(x - pad))
        top = max(0, int(y - pad))
        right = min(frame_w, int(x + w + pad))
        bottom = min(frame_h, int(y + h + pad))

        cropped = img.crop((left, top, right, bottom))
        buf = io.BytesIO()
        cropped.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    def get_temporal_crops(self) -> list[tuple[bytes, float]]:
        """Return top-K crops for temporal embedding aggregation."""
        if self.temporal_crops:
            return self.temporal_crops
        if self.best_crop_bytes:
            return [(self.best_crop_bytes, self.best_quality)]
        return []

    def should_capture_live(self, min_stable: int = 8, min_frames: int = 10) -> bool:
        """Live mode: capture when stable and quality is good enough."""
        return (
            not self.captured
            and self.frames_seen >= min_frames
            and self.stable_count >= min_stable
            and self.best_quality > 0.3
        )

    def should_capture_video(self, min_frames: int = 5) -> bool:
        """Video mode: capture if seen enough frames (quality already tracked)."""
        return self.frames_seen >= min_frames and len(self.best_crop_bytes) > 0


class ByteTracker:
    """
    ByteTrack multi-object tracker.

    Two-pass matching:
    1. High-confidence detections matched to active tracks (IoU > threshold)
    2. Low-confidence detections matched to remaining tracks (tighter threshold)
    3. Unmatched high-conf detections become new tracks
    4. Unmatched tracks incremented as lost; removed after max_lost frames
    """

    def __init__(
        self,
        high_conf_threshold: float = 0.5,
        iou_threshold: float = 0.3,
        low_conf_iou_threshold: float = 0.5,
        max_lost: int = 30,
    ):
        self.high_conf_threshold = high_conf_threshold
        self.iou_threshold = iou_threshold
        self.low_conf_iou_threshold = low_conf_iou_threshold
        self.max_lost = max_lost
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[dict],
        frame_bytes: bytes = None,
        frame_w: int = 0,
        frame_h: int = 0,
    ) -> list[Track]:
        """
        Update tracker with new frame detections.

        Args:
            detections: list of {"bbox": (x,y,w,h), "confidence": float, "class_id": int}
            frame_bytes: JPEG bytes of the full frame (for quality scoring)
            frame_w, frame_h: frame dimensions

        Returns:
            All active tracks (including newly created ones)
        """
        high_conf = [d for d in detections if d["confidence"] >= self.high_conf_threshold]
        low_conf = [d for d in detections if d["confidence"] < self.high_conf_threshold]

        active_tracks = [t for t in self.tracks if t.frames_lost == 0]
        lost_tracks = [t for t in self.tracks if t.frames_lost > 0]

        # Pass 1: match high-confidence detections to active tracks
        if active_tracks and high_conf:
            cost = compute_iou_matrix(active_tracks, high_conf)
            matches, unmatched_track_idx, unmatched_det_idx = hungarian_match(
                cost, 1.0 - self.iou_threshold
            )

            for t_idx, d_idx in matches:
                active_tracks[t_idx].update(high_conf[d_idx], frame_bytes, frame_w, frame_h)

            remaining_tracks = [active_tracks[i] for i in unmatched_track_idx]
            remaining_dets_high = [high_conf[i] for i in unmatched_det_idx]
        else:
            remaining_tracks = active_tracks
            remaining_dets_high = high_conf

        # Pass 2: match low-confidence detections to remaining active tracks
        if remaining_tracks and low_conf:
            cost2 = compute_iou_matrix(remaining_tracks, low_conf)
            matches2, still_unmatched_tracks, _ = hungarian_match(
                cost2, 1.0 - self.low_conf_iou_threshold
            )

            for t_idx, d_idx in matches2:
                remaining_tracks[t_idx].update(low_conf[d_idx], frame_bytes, frame_w, frame_h)

            final_unmatched = [remaining_tracks[i] for i in still_unmatched_tracks]
        else:
            final_unmatched = remaining_tracks

        # Pass 3: try to re-activate lost tracks with unmatched high-conf detections
        if lost_tracks and remaining_dets_high:
            cost3 = compute_iou_matrix(lost_tracks, remaining_dets_high)
            matches3, _, still_unmatched_dets = hungarian_match(
                cost3, 1.0 - self.iou_threshold
            )

            for t_idx, d_idx in matches3:
                lost_tracks[t_idx].update(remaining_dets_high[d_idx], frame_bytes, frame_w, frame_h)

            remaining_dets_high = [remaining_dets_high[i] for i in still_unmatched_dets]

        # Create new tracks for unmatched high-confidence detections
        for det in remaining_dets_high:
            new_track = Track(
                track_id=self._next_id,
                bbox=det["bbox"],
                confidence=det["confidence"],
            )
            if frame_bytes and frame_w > 0:
                new_track.update(det, frame_bytes, frame_w, frame_h)
            self.tracks.append(new_track)
            self._next_id += 1

        # Increment lost counter for unmatched tracks
        for track in final_unmatched:
            track.frames_lost += 1

        # Remove tracks lost for too long
        self.tracks = [t for t in self.tracks if t.frames_lost < self.max_lost]

        return [t for t in self.tracks if t.frames_lost == 0]

    def get_capturable_live(self) -> list[Track]:
        """Return tracks ready for live auto-capture."""
        ready = [t for t in self.tracks if t.should_capture_live()]
        for t in ready:
            t.captured = True
        return ready

    def get_all_results_video(self, min_frames: int = 5) -> list[Track]:
        """Return all tracks with enough frames for video mode."""
        return [t for t in self.tracks if t.should_capture_video(min_frames)]


# ─── YOLO Detector ─────────────────────────────────────────────────────────────

# COCO classes that are relevant for clothing/fashion detection
CLOTHING_CLASSES = {
    0,   # person (we detect person then crop sub-regions)
    24,  # backpack
    25,  # umbrella
    26,  # handbag
    27,  # tie
    28,  # suitcase
}

# For a general approach: accept all detections and let downstream
# vision pipeline (Florence) validate if it's clothing
PERMISSIVE_MODE = True


class YOLODetector:
    """
    YOLO-based object detector using ultralytics.
    Loads YOLOv8n (nano) for fast inference.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.4):
        self.confidence = confidence
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            logger.info(f"YOLO model loaded: {self._model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run detection on a single frame (numpy BGR array).

        Returns list of {"bbox": (x,y,w,h), "confidence": float, "class_id": int}
        """
        self._load_model()

        results = self._model(frame, conf=self.confidence, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if not PERMISSIVE_MODE and cls_id not in CLOTHING_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "confidence": conf,
                    "class_id": cls_id,
                })

        return detections

    def detect_from_bytes(self, image_bytes: bytes) -> list[dict]:
        """Run detection on JPEG/PNG bytes."""
        import cv2
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return []
        return self.detect(frame)


# ─── Temporal Embedding Aggregation ────────────────────────────────────────────

def aggregate_temporal_embeddings(
    crops: list[tuple[bytes, float]],
) -> list[float] | None:
    """
    Segment + embed multiple crops of the same tracked object and produce a single
    quality-weighted aggregated embedding.

    Pipeline per crop: segment (remove background) → embed (FashionCLIP)
    Then: quality-weighted average across all crops → L2 normalize.

    This produces a more robust, background-free representation than any single frame.

    Returns None if embedding fails for all crops.
    """
    from services.fashion_clip import embed_image
    from services.segmentation import segment_for_embedding

    embeddings = []
    weights = []

    for crop_bytes, quality in crops:
        try:
            segmented = segment_for_embedding(crop_bytes)
            emb = embed_image(segmented)
            embeddings.append(np.array(emb, dtype=np.float32))
            weights.append(quality)
        except Exception as e:
            logger.warning(f"Temporal embed failed for one crop: {e}")
            continue

    if not embeddings:
        return None

    if len(embeddings) == 1:
        return embeddings[0].tolist()

    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum()

    aggregated = np.zeros_like(embeddings[0])
    for emb, w in zip(embeddings, weights):
        aggregated += w * emb

    norm = np.linalg.norm(aggregated)
    if norm > 0:
        aggregated = aggregated / norm

    logger.debug(f"Temporal aggregation: {len(embeddings)} embeddings -> 1 (weights: {weights.tolist()})")
    return aggregated.tolist()


# ─── Video Processing Pipeline ─────────────────────────────────────────────────

@dataclass
class CropResult:
    """Result from video processing: one best crop per tracked object."""
    track_id: int
    crop_bytes: bytes
    quality_score: float
    frames_seen: int
    temporal_embedding: list | None = None  # Aggregated embedding from multiple frames


def process_video_frames(
    frames: list[tuple[bytes, int, int]],
    fps: float = 3.0,
    min_frames: int = 5,
    compute_embeddings: bool = True,
) -> list[CropResult]:
    """
    Process video frames through YOLO + ByteTrack pipeline.
    Returns one best crop per unique tracked object with temporally aggregated embeddings.

    Args:
        frames: list of (jpeg_bytes, width, height) tuples
        fps: frames per second the video was sampled at
        min_frames: minimum frames an object must appear in to be kept
        compute_embeddings: if True, compute temporal embedding aggregation per track

    Returns:
        List of CropResult with aggregated embeddings
    """
    detector = YOLODetector()
    tracker = ByteTracker(max_lost=int(fps * 2))

    for frame_bytes, w, h in frames:
        detections = detector.detect_from_bytes(frame_bytes)
        tracker.update(detections, frame_bytes, w, h)

    results = tracker.get_all_results_video(min_frames=min_frames)

    logger.info(
        f"Video tracking: {len(frames)} frames -> "
        f"{len(tracker.tracks)} total tracks -> "
        f"{len(results)} items kept"
    )

    crop_results = []
    for track in results:
        if not track.best_crop_bytes:
            continue

        temporal_emb = None
        if compute_embeddings:
            temporal_crops = track.get_temporal_crops()
            if temporal_crops:
                temporal_emb = aggregate_temporal_embeddings(temporal_crops)

        crop_results.append(CropResult(
            track_id=track.track_id,
            crop_bytes=track.best_crop_bytes,
            quality_score=track.best_quality,
            frames_seen=track.frames_seen,
            temporal_embedding=temporal_emb,
        ))

    return crop_results
