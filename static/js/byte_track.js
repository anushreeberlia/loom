/**
 * ByteTrack multi-object tracker (JavaScript implementation).
 *
 * Tracks objects across video frames using IoU-based association.
 * Two-pass matching: high-confidence first, then low-confidence to remaining tracks.
 */

function computeIoU(boxA, boxB) {
  const [ax, ay, aw, ah] = boxA;
  const [bx, by, bw, bh] = boxB;

  const ax2 = ax + aw, ay2 = ay + ah;
  const bx2 = bx + bw, by2 = by + bh;

  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(ax, bx));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(ay, by));
  const intersection = ix * iy;

  const union = aw * ah + bw * bh - intersection;
  if (union <= 0) return 0;
  return intersection / union;
}

function computeIoUMatrix(tracks, detections) {
  const matrix = [];
  for (let i = 0; i < tracks.length; i++) {
    matrix[i] = [];
    for (let j = 0; j < detections.length; j++) {
      matrix[i][j] = 1.0 - computeIoU(tracks[i].bbox, detections[j].bbox);
    }
  }
  return matrix;
}

function greedyMatch(costMatrix, threshold) {
  const nTracks = costMatrix.length;
  const nDets = nTracks > 0 ? costMatrix[0].length : 0;

  if (nTracks === 0 || nDets === 0) {
    return {
      matches: [],
      unmatchedTracks: Array.from({ length: nTracks }, (_, i) => i),
      unmatchedDets: Array.from({ length: nDets }, (_, i) => i),
    };
  }

  const entries = [];
  for (let i = 0; i < nTracks; i++) {
    for (let j = 0; j < nDets; j++) {
      entries.push({ i, j, cost: costMatrix[i][j] });
    }
  }
  entries.sort((a, b) => a.cost - b.cost);

  const matches = [];
  const usedTracks = new Set();
  const usedDets = new Set();

  for (const { i, j, cost } of entries) {
    if (cost > threshold) break;
    if (usedTracks.has(i) || usedDets.has(j)) continue;
    matches.push([i, j]);
    usedTracks.add(i);
    usedDets.add(j);
  }

  const unmatchedTracks = [];
  const unmatchedDets = [];
  for (let i = 0; i < nTracks; i++) if (!usedTracks.has(i)) unmatchedTracks.push(i);
  for (let j = 0; j < nDets; j++) if (!usedDets.has(j)) unmatchedDets.push(j);

  return { matches, unmatchedTracks, unmatchedDets };
}

class TrackedObject {
  constructor(trackId, bbox, confidence) {
    this.trackId = trackId;
    this.bbox = bbox;
    this.confidence = confidence;
    this.framesSeen = 1;
    this.framesLost = 0;
    this.stableCount = 0;
    this.lastCenter = null;
    this.captured = false;

    this.bestBlob = null;
    this.bestScore = 0;

    this._centerThreshold = 0.08;
  }

  update(detection) {
    this.bbox = detection.bbox;
    this.confidence = detection.confidence;
    this.framesSeen++;
    this.framesLost = 0;

    const [x, y, w, h] = this.bbox;
    const center = { x: x + w / 2, y: y + h / 2 };

    if (this.lastCenter) {
      const dx = Math.abs(center.x - this.lastCenter.x) / (this._frameW || 1);
      const dy = Math.abs(center.y - this.lastCenter.y) / (this._frameH || 1);
      if (dx < this._centerThreshold && dy < this._centerThreshold) {
        this.stableCount++;
      } else {
        this.stableCount = 0;
      }
    }
    this.lastCenter = center;
  }

  setFrameDimensions(w, h) {
    this._frameW = w;
    this._frameH = h;
  }

  shouldCapture(minStable = 5, minFrames = 8, qualityThreshold = 0.3) {
    return (
      !this.captured &&
      this.framesSeen >= minFrames &&
      this.stableCount >= minStable &&
      this.bestScore > qualityThreshold
    );
  }

  markCaptured() {
    this.captured = true;
  }
}

class ByteTracker {
  constructor(options = {}) {
    this.highConfThreshold = options.highConfThreshold || 0.5;
    this.iouThreshold = options.iouThreshold || 0.3;
    this.lowConfIouThreshold = options.lowConfIouThreshold || 0.5;
    this.maxLost = options.maxLost || 30;
    this.tracks = [];
    this._nextId = 1;
  }

  update(detections) {
    const highConf = detections.filter(d => d.confidence >= this.highConfThreshold);
    const lowConf = detections.filter(d => d.confidence < this.highConfThreshold);

    const activeTracks = this.tracks.filter(t => t.framesLost === 0);
    const lostTracks = this.tracks.filter(t => t.framesLost > 0);

    let remainingTracks = activeTracks;
    let remainingDetsHigh = highConf;

    // Pass 1: match high-confidence to active tracks
    if (activeTracks.length > 0 && highConf.length > 0) {
      const cost = computeIoUMatrix(activeTracks, highConf);
      const { matches, unmatchedTracks, unmatchedDets } = greedyMatch(cost, 1.0 - this.iouThreshold);

      for (const [tIdx, dIdx] of matches) {
        activeTracks[tIdx].update(highConf[dIdx]);
      }

      remainingTracks = unmatchedTracks.map(i => activeTracks[i]);
      remainingDetsHigh = unmatchedDets.map(i => highConf[i]);
    }

    // Pass 2: match low-confidence to remaining active tracks
    if (remainingTracks.length > 0 && lowConf.length > 0) {
      const cost2 = computeIoUMatrix(remainingTracks, lowConf);
      const { matches: matches2, unmatchedTracks: stillUnmatched } = greedyMatch(cost2, 1.0 - this.lowConfIouThreshold);

      for (const [tIdx, dIdx] of matches2) {
        remainingTracks[tIdx].update(lowConf[dIdx]);
      }

      remainingTracks = stillUnmatched.map(i => remainingTracks[i]);
    }

    // Pass 3: re-activate lost tracks with unmatched high-conf detections
    if (lostTracks.length > 0 && remainingDetsHigh.length > 0) {
      const cost3 = computeIoUMatrix(lostTracks, remainingDetsHigh);
      const { matches: matches3, unmatchedDets: stillUnmatchedDets } = greedyMatch(cost3, 1.0 - this.iouThreshold);

      for (const [tIdx, dIdx] of matches3) {
        lostTracks[tIdx].update(remainingDetsHigh[dIdx]);
      }

      remainingDetsHigh = stillUnmatchedDets.map(i => remainingDetsHigh[i]);
    }

    // Create new tracks for unmatched high-confidence detections
    for (const det of remainingDetsHigh) {
      const track = new TrackedObject(this._nextId++, det.bbox, det.confidence);
      this.tracks.push(track);
    }

    // Increment lost counter for unmatched active tracks
    for (const track of remainingTracks) {
      track.framesLost++;
    }

    // Remove tracks lost for too long
    this.tracks = this.tracks.filter(t => t.framesLost < this.maxLost);

    return this.tracks.filter(t => t.framesLost === 0);
  }

  getCapturable() {
    const ready = this.tracks.filter(t => t.shouldCapture());
    for (const t of ready) t.markCaptured();
    return ready;
  }

  reset() {
    this.tracks = [];
    this._nextId = 1;
  }
}

// Export for use in inventory.html
window.ByteTracker = ByteTracker;
window.TrackedObject = TrackedObject;
window.computeIoU = computeIoU;
