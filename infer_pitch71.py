"""
infer_pitch71.py
================
Inference pipeline — trajectory-based.

WHAT CHANGED FROM THE OLD VERSION
---------------------------------
The old pipeline used audio to guess WHEN impact happened, then scanned a
small window of frames around that guess looking for the ball. That was the
weak link: the audio impact detectors were frequently confident and wrong,
and everything downstream inherited the error.

This version finds impact from the ball's actual flight instead:
    1. Run the fine-tuned ball detector across EVERY frame of the clip.
    2. Score the detections and keep only the ones forming a physically
       coherent flight path (path_scoring.analyze_pitch).
    3. Impact = the LAST point of that validated path.

The two CLASSIFIERS are kept exactly as they were, because deciding
mitt-vs-bat was never the problem:
    - audio_classifier      (bat vs mitt from sound)
    - visual_classifier     (bat vs mitt from a frame)

The two audio IMPACT DETECTORS are gone entirely, along with the
peak-finding, weighted-average fallback, and the ±0.3s YOLO scan window.

Usage:
    python infer_pitch71.py --video path/to/pitch.mov
    python infer_pitch71.py --video path/to/pitch.mov --debug
"""

import os, sys, argparse, subprocess, tempfile, wave, struct, time, json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from ultralytics import YOLO
import cv2

from path_scoring import analyze_pitch

# ── MODEL PATHS ───────────────────────────────────────────────────────────────
import os as _os
# BASE can be overridden with the ZONEARC_MODELS_DIR environment variable —
# this lets the exact same file run unmodified on your Windows laptop AND
# inside a Linux server/container, without hand-editing paths in two places.
BASE = _os.environ.get("ZONEARC_MODELS_DIR", "./models")

MODELS = {
    # classifiers — kept
    "audio_classifier":  BASE + "/Models/Audio_Impact/audio_classifier.pt",
    "visual_classifier": BASE + "/Models/Visual_Impact/classifier_mitt_vs_bat.pt",
    # the fine-tuned ball detector — this replaces yolo_mitt / yolo_bat
    "ball_detector":     BASE + "/Models/Visual_Impact/best.pt",
}

# ── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
CLIP_SEC     = 0.5
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SEC)
STRIDE_SEC   = 0.15    # was 0.05 — this only needs to find roughly where the
                        # clearest bat/mitt sound is, not pinpoint timing, so
                        # a coarser scan cuts audio-classifier calls by ~3x
                        # with no real accuracy cost.
IMG_SIZE     = 224
BALL_CONF    = 0.15    # low threshold on purpose — path scoring filters the noise
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioCNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=64, stride=4, padding=32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=16), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=16, stride=4, padding=8), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=4), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(256 * 16, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1)))


print("Loading models...")

audio_clf = AudioCNN(output_size=2).to(device)
audio_clf.load_state_dict(torch.load(MODELS["audio_classifier"], map_location=device)["model_state_dict"])
audio_clf.eval()

visual_clf_model = models.resnet18(weights=None)
visual_clf_model.fc = nn.Linear(visual_clf_model.fc.in_features, 2)
ckpt = torch.load(MODELS["visual_classifier"], map_location=device)
visual_clf_model.load_state_dict(ckpt["model_state_dict"])
visual_clf_model = visual_clf_model.to(device)
visual_clf_model.eval()

ball_detector = YOLO(MODELS["ball_detector"])
print(f"Ball detector device: {ball_detector.device}")
if str(ball_detector.device) == "cpu":
    print("  WARNING: running on CPU — this is almost certainly why frame scanning is slow.")

print("All models loaded.")


# ── audio helpers (classifier only) ───────────────────────────────────────────
def extract_audio(video_path):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn",
         "-acodec", "pcm_s16le", "-ar", str(SAMPLE_RATE), "-ac", "1", tmp.name],
        capture_output=True)
    if r.returncode != 0 or os.path.getsize(tmp.name) < 1000:
        try: os.unlink(tmp.name)
        except: pass
        return None
    with wave.open(tmp.name, "rb") as wf:
        n_ch, sw, nf = wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
        raw = wf.readframes(nf)
    fmt = {1: "b", 2: "h", 4: "i"}[sw]
    samples = np.array(struct.unpack(f"{nf*n_ch}{fmt}", raw), dtype=np.float32)
    if n_ch == 2:
        samples = samples[::2]
    samples /= (2 ** (sw * 8 - 1))
    os.unlink(tmp.name)
    return samples


def load_clip(samples, start_idx):
    clip = samples[start_idx:start_idx + CLIP_SAMPLES]
    t = torch.tensor(clip)
    if t.shape[0] < CLIP_SAMPLES:
        t = torch.nn.functional.pad(t, (0, CLIP_SAMPLES - t.shape[0]))
    return t


def get_audio_classifier_probs(samples):
    stride = int(STRIDE_SEC * SAMPLE_RATE)
    best_conf, best_probs = -1, None
    for start in range(0, max(1, len(samples) - CLIP_SAMPLES), stride):
        x = load_clip(samples, start).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(audio_clf(x), dim=1)[0].cpu().numpy()
        conf = float(np.max(probs))
        if conf > best_conf:
            best_conf, best_probs = conf, probs
    return best_probs


# ── visual classifier ─────────────────────────────────────────────────────────
val_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_visual_classifier_probs(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = val_tf(rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.softmax(visual_clf_model(x), dim=1)[0].cpu().numpy()


# ── ball tracking across the whole clip ───────────────────────────────────────
BALL_DETECT_STRIDE = 4   # run detection on every Nth frame instead of every
                          # single one. Path scoring tolerates gaps up to 5
                          # frames, so stride=4 still leaves a comfortable
                          # margin while cutting the number of YOLO calls to
                          # roughly a quarter of the full frame count.
BALL_BATCH_SIZE = 8      # frames per YOLO call — batching cuts per-call
                          # overhead versus one frame at a time (helps more
                          # on GPU than CPU, but still nonzero either way).
BALL_IMGSZ = 640          # inference resolution passed to YOLO. Lowering this
                          # (e.g. 480) roughly trades accuracy for speed on
                          # CPU — only drop this if stride alone isn't enough,
                          # and re-validate against known-good clips after.


def scan_all_frames(video_path, debug=False):
    """Run the fine-tuned ball detector across the clip, batched and
    strided for speed. Returns (detections, width, height, fps,
    total_frames, mid_frame). Detection frame indices are the REAL frame
    numbers in the source video, not batch/loop counters — gap logic and
    the reported impact_frame stay correct regardless of the stride."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mid_frame = None
    frames_to_check = []  # list of (real_frame_idx, frame_image)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if mid_frame is None and idx > 0 and idx % 30 == 0:
            mid_frame = frame.copy()
        if idx % BALL_DETECT_STRIDE == 0:
            frames_to_check.append((idx, frame))
        idx += 1
    cap.release()
    total_frames = idx

    dets = []
    for batch_start in range(0, len(frames_to_check), BALL_BATCH_SIZE):
        batch = frames_to_check[batch_start:batch_start + BALL_BATCH_SIZE]
        batch_indices = [b[0] for b in batch]
        batch_images = [b[1] for b in batch]

        results = ball_detector(batch_images, verbose=False, conf=BALL_CONF, imgsz=BALL_IMGSZ)
        for real_idx, r in zip(batch_indices, results):
            if len(r.boxes) > 0:
                b = r.boxes[r.boxes.conf.argmax()]
                cx, cy = b.xywh[0][:2].tolist()
                dets.append((real_idx, int(cx), int(cy), float(b.conf)))
                if debug:
                    print(f"    frame {real_idx}: ball ({int(cx)},{int(cy)}) conf={float(b.conf):.3f}")

    return dets, w, h, fps, total_frames, mid_frame


def infer(video_path, debug=False):
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    t0 = time.perf_counter()

    # ── 1. track the ball across the entire clip ──────────────────────────
    print("  Scanning all frames for the ball...")
    t_scan_start = time.perf_counter()
    dets, w, h, fps, total_frames, mid_frame = scan_all_frames(video_path, debug=debug)
    t_scan = time.perf_counter() - t_scan_start
    print(f"  Video: {w}x{h} {fps:.1f}fps {total_frames} frames — {len(dets)} raw detections")
    print(f"  [TIMING] ball scan: {t_scan:.2f}s  ({t_scan/max(total_frames,1)*1000:.0f}ms/frame avg)")

    # ── 2. classify mitt vs bat (unchanged) ───────────────────────────────
    t_audio_start = time.perf_counter()
    samples = extract_audio(video_path)
    if samples is not None:
        audio_probs = get_audio_classifier_probs(samples)
        audio_bat, audio_mitt = float(audio_probs[0]), float(audio_probs[1])
        print(f"  Audio classifier  → bat={audio_bat:.3f}  mitt={audio_mitt:.3f}")
    else:
        audio_bat = audio_mitt = 0.5
        print("  Audio classifier  → no audio track, skipped")
    t_audio = time.perf_counter() - t_audio_start
    print(f"  [TIMING] audio classifier: {t_audio:.2f}s")

    t_visual_start = time.perf_counter()
    if mid_frame is not None:
        visual_probs = get_visual_classifier_probs(mid_frame)
        visual_bat, visual_mitt = float(visual_probs[0]), float(visual_probs[1])
        print(f"  Visual classifier → bat={visual_bat:.3f}  mitt={visual_mitt:.3f}")
        combined_bat = audio_bat * visual_bat
        combined_mitt = audio_mitt * visual_mitt
    else:
        combined_bat, combined_mitt = audio_bat, audio_mitt
        print("  Visual classifier → no frame available, audio only")
    t_visual = time.perf_counter() - t_visual_start
    print(f"  [TIMING] visual classifier: {t_visual:.2f}s")

    impact_type = "mitt" if combined_mitt > combined_bat else "bat"
    combined_conf = max(combined_mitt, combined_bat)
    print(f"  Combined → {impact_type.upper()} (conf={combined_conf:.4f})")

    # ── 3. find the real flight path and where it ends ────────────────────
    result = analyze_pitch(dets, w, h)
    s = result["summary"]
    impact = result["impact"]

    print(f"  Flight path: {s['n_path']} points kept, {s['n_rejected']} rejected "
          f"(curve rmse={s['rmse']})")
    if debug:
        for row in result["rows"]:
            if row["status"] == "rejected":
                print(f"    rejected frame {row['frame']} ({row['x']},{row['y']}): {row['reason']}")

    if impact is None:
        print("  No valid flight path found — cannot determine impact")
        return {
            "video": os.path.basename(video_path),
            "impact_type": impact_type,
            "impact_ts": None, "impact_frame": None,
            "ball_x": None, "ball_y": None,
            "combined_conf": round(combined_conf, 4),
            "path_points": 0, "path_rmse": None,
            "yolo_conf": None,
            "method": "no_path",
        }

    impact_frame, ball_x, ball_y, ball_conf = impact
    impact_ts = impact_frame / fps if fps else 0.0

    print(f"  Impact: frame {impact_frame} (t={impact_ts:.3f}s)")
    print(f"  Ball:   x={ball_x}  y={ball_y}  (conf={ball_conf:.3f})")
    print(f"  Total time: {time.perf_counter()-t0:.2f}s")

    # full flight path, normalized to 0-1 so it scales correctly no matter
    # what size the video is displayed at later
    full_path = [
        {"frame": d[0], "x": round(d[1] / w, 5), "y": round(d[2] / h, 5), "conf": round(d[3], 3)}
        for d in result["path"]
    ]

    return {
        "video":         os.path.basename(video_path),
        "impact_type":   impact_type,
        "impact_ts":     round(impact_ts, 4),
        "impact_frame":  impact_frame,
        "ball_x":        ball_x,
        "ball_y":        ball_y,
        "combined_conf": round(combined_conf, 4),
        "path_points":   s["n_path"],
        "path_rmse":     s["rmse"],
        "yolo_conf":     round(ball_conf, 4),
        "method":        "trajectory",
        "frame_width":   w,
        "frame_height":  h,
        "fps":           fps,
        "flight_path":   full_path,   # list of {frame, x, y, conf} — x/y normalized 0-1
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch impact inference (trajectory-based)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        sys.exit(1)

    result = infer(args.video, debug=args.debug)
    if result:
        print(f"\nResult: {result}")
