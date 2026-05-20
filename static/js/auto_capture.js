/**
 * Auto-capture logic for live scan mode.
 *
 * Integrates YOLO ONNX detection with ByteTrack to provide
 * per-object quality scoring and automatic capture when stable.
 */

class AutoCaptureManager {
  constructor(options = {}) {
    this.tracker = new ByteTracker({
      highConfThreshold: options.highConfThreshold || 0.45,
      iouThreshold: 0.3,
      maxLost: 30,
    });

    this.minStableFrames = options.minStableFrames || 8;
    this.minTrackFrames = options.minTrackFrames || 10;
    this.qualityThreshold = options.qualityThreshold || 0.3;
    this.cooldownMs = options.cooldownMs || 1500;

    this._lastCaptureTime = 0;
    this._session = null;
    this._onCapture = options.onCapture || (() => {});
    this._onTrackUpdate = options.onTrackUpdate || (() => {});
  }

  /**
   * Process detections from one frame.
   * Returns active tracks with stability info for rendering.
   */
  processFrame(detections, frameW, frameH) {
    const activeTracks = this.tracker.update(detections);

    for (const track of activeTracks) {
      track.setFrameDimensions(frameW, frameH);
    }

    this._onTrackUpdate(activeTracks);

    return activeTracks;
  }

  /**
   * Check and trigger captures. Call AFTER quality scores are updated.
   */
  checkCaptures() {
    const now = Date.now();
    if (now - this._lastCaptureTime < this.cooldownMs) {
      return;
    }

    const ready = this.tracker.tracks.filter(t =>
      t.shouldCapture(this.minStableFrames, this.minTrackFrames, this.qualityThreshold)
    );
    if (ready.length > 0) {
      this._lastCaptureTime = now;
      for (const track of ready) {
        track.markCaptured();
        this._onCapture(track);
      }
    }
  }

  /**
   * Update best frame quality for a tracked object.
   * Call this with a cropped blob + quality score.
   */
  updateBestFrame(trackId, blob, qualityScore) {
    const track = this.tracker.tracks.find(t => t.trackId === trackId);
    if (track && qualityScore > track.bestScore) {
      track.bestBlob = blob;
      track.bestScore = qualityScore;
    }
  }

  reset() {
    this.tracker.reset();
    this._lastCaptureTime = 0;
  }
}

/**
 * Canvas-based quality scoring for browser frames.
 * Approximates sharpness using Laplacian-like kernel.
 */
function computeFrameQuality(canvas, bbox, frameW, frameH) {
  const [x, y, w, h] = bbox;

  // Centeredness (0-1, higher = more centered)
  const cx = (x + w / 2) / frameW;
  const cy = (y + h / 2) / frameH;
  const dist = Math.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2);
  const centeredness = Math.max(0, 1.0 - dist * 2);

  // Size ratio (prefer larger objects)
  const areaRatio = Math.min((w * h) / (frameW * frameH) * 5, 1.0);

  // Sharpness via variance of grayscale differences
  let sharpness = 0.5; // default if we can't compute
  try {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    const cropX = Math.max(0, Math.round(x));
    const cropY = Math.max(0, Math.round(y));
    const cropW = Math.min(Math.round(w), frameW - cropX);
    const cropH = Math.min(Math.round(h), frameH - cropY);

    if (cropW > 4 && cropH > 4) {
      const imageData = ctx.getImageData(cropX, cropY, cropW, cropH);
      const data = imageData.data;

      // Subsample for speed (every 4th pixel)
      let sum = 0, sumSq = 0, count = 0;
      for (let i = 0; i < data.length; i += 16) {
        const gray = data[i] * 0.299 + data[i + 1] * 0.587 + data[i + 2] * 0.114;
        sum += gray;
        sumSq += gray * gray;
        count++;
      }
      if (count > 0) {
        const mean = sum / count;
        const variance = (sumSq / count) - (mean * mean);
        sharpness = Math.min(variance / 2000, 1.0);
      }
    }
  } catch (e) {
    // canvas tainted or other issue -- use default
  }

  return sharpness * 0.5 + centeredness * 0.3 + areaRatio * 0.2;
}

/**
 * Crop a bounding box region from video element to a Blob.
 */
function cropBboxToBlob(video, bbox, padding = 0.12) {
  return new Promise((resolve) => {
    const [x, y, w, h] = bbox;
    const pad = Math.max(w, h) * padding;
    const cropX = Math.max(0, x - pad);
    const cropY = Math.max(0, y - pad);
    const cropW = Math.min(video.videoWidth - cropX, w + 2 * pad);
    const cropH = Math.min(video.videoHeight - cropY, h + 2 * pad);

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = cropW;
    tempCanvas.height = cropH;
    tempCanvas.getContext('2d').drawImage(
      video, cropX, cropY, cropW, cropH, 0, 0, cropW, cropH
    );

    tempCanvas.toBlob(resolve, 'image/jpeg', 0.92);
  });
}

/**
 * YOLO ONNX inference wrapper for browser.
 * Uses ONNX Runtime Web to run YOLOv8-nano.
 */
class YOLODetectorBrowser {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.session = null;
    this.inputSize = 640;
    this.confThreshold = 0.4;
    this.iouThreshold = 0.45;
    this._loading = null;
  }

  async load() {
    if (this.session) return true;
    if (this._loading) return this._loading;

    this._loading = (async () => {
      try {
        if (!window.ort) {
          await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.min.js');
          if (window.ort) {
            window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/';
          }
        }
        console.log('[AutoCapture] ONNX Runtime loaded, creating session for:', this.modelUrl);
        this.session = await window.ort.InferenceSession.create(this.modelUrl, {
          executionProviders: ['webgl', 'wasm'],
        });
        console.log('[AutoCapture] YOLO model loaded successfully');
        return true;
      } catch (e) {
        console.error('[AutoCapture] YOLO ONNX load failed:', e);
        return false;
      }
    })();

    return this._loading;
  }

  async detect(videoElement) {
    if (!this.session) return [];

    const canvas = document.createElement('canvas');
    canvas.width = this.inputSize;
    canvas.height = this.inputSize;
    const ctx = canvas.getContext('2d');

    // Letterbox resize
    const vw = videoElement.videoWidth;
    const vh = videoElement.videoHeight;
    const scale = Math.min(this.inputSize / vw, this.inputSize / vh);
    const nw = Math.round(vw * scale);
    const nh = Math.round(vh * scale);
    const dx = (this.inputSize - nw) / 2;
    const dy = (this.inputSize - nh) / 2;

    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, this.inputSize, this.inputSize);
    ctx.drawImage(videoElement, dx, dy, nw, nh);

    const imageData = ctx.getImageData(0, 0, this.inputSize, this.inputSize);
    const input = this._preprocess(imageData);

    try {
      const feeds = { images: input };
      const results = await this.session.run(feeds);
      const output = results[Object.keys(results)[0]];
      return this._postprocess(output, vw, vh, scale, dx, dy);
    } catch (e) {
      console.warn('YOLO inference error:', e);
      return [];
    }
  }

  _preprocess(imageData) {
    const { data, width, height } = imageData;
    const float32 = new Float32Array(3 * width * height);
    const size = width * height;

    for (let i = 0; i < size; i++) {
      float32[i] = data[i * 4] / 255.0;           // R
      float32[size + i] = data[i * 4 + 1] / 255.0; // G
      float32[2 * size + i] = data[i * 4 + 2] / 255.0; // B
    }

    return new window.ort.Tensor('float32', float32, [1, 3, height, width]);
  }

  _postprocess(output, origW, origH, scale, dx, dy) {
    const data = output.data;
    const [batch, features, numBoxes] = output.dims;
    const numClasses = features - 4;
    const detections = [];

    for (let i = 0; i < numBoxes; i++) {
      const cx = data[0 * numBoxes + i];
      const cy = data[1 * numBoxes + i];
      const w = data[2 * numBoxes + i];
      const h = data[3 * numBoxes + i];

      let maxConf = 0;
      let classId = 0;
      for (let c = 0; c < numClasses; c++) {
        const conf = data[(4 + c) * numBoxes + i];
        if (conf > maxConf) {
          maxConf = conf;
          classId = c;
        }
      }

      if (maxConf < this.confThreshold) continue;

      // Convert from letterboxed coords to original image coords
      const x1 = (cx - w / 2 - dx) / scale;
      const y1 = (cy - h / 2 - dy) / scale;
      const bw = w / scale;
      const bh = h / scale;

      detections.push({
        bbox: [
          Math.max(0, x1),
          Math.max(0, y1),
          Math.min(bw, origW - x1),
          Math.min(bh, origH - y1),
        ],
        confidence: maxConf,
        classId: classId,
      });
    }

    return this._nms(detections);
  }

  _nms(detections) {
    detections.sort((a, b) => b.confidence - a.confidence);
    const kept = [];

    for (const det of detections) {
      let dominated = false;
      for (const k of kept) {
        if (computeIoU(det.bbox, k.bbox) > this.iouThreshold) {
          dominated = true;
          break;
        }
      }
      if (!dominated) kept.push(det);
    }

    return kept;
  }
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
    const s = document.createElement('script');
    s.src = src;
    s.onload = resolve;
    s.onerror = reject;
    document.head.appendChild(s);
  });
}

// Export
window.AutoCaptureManager = AutoCaptureManager;
window.YOLODetectorBrowser = YOLODetectorBrowser;
window.computeFrameQuality = computeFrameQuality;
window.cropBboxToBlob = cropBboxToBlob;
