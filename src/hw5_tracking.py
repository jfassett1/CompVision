import os
import tempfile
import cv2
import numpy as np

# Ensure results directory exists
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================
# Utility: decode uploaded video
# ============================================
def decode_video_file(file_storage_or_path, max_frames=None):
    if hasattr(file_storage_or_path, "read"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file_storage_or_path.read())
        tmp.flush()
        tmp.close()
        path = tmp.name
    else:
        path = file_storage_or_path

    cap = cv2.VideoCapture(path)
    frames = []
    count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        count += 1
        if max_frames and count >= max_frames:
            break

    cap.release()
    return frames


# ============================================
# Utility: Save AVI (MJPG) — stable everywhere
# ============================================
def save_as_avi(frames, prefix, fps=15):
    if not frames:
        return None

    h, w = frames[0].shape[:2]

    rel = f"results/{prefix}.avi"
    abs_path = os.path.join("static", rel)

    # MJPG is universally stable and browser-compatible
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(abs_path, fourcc, fps, (w, h))

    for f in frames:
        writer.write(f)

    writer.release()
    return rel


# ============================================
# MARKER-BASED TRACKER (QR only)
# ============================================
def track_marker(video_frames, save_prefix="hw5_marker"):
    detector = cv2.QRCodeDetector()
    processed = []
    results = []

    for idx, frame in enumerate(video_frames):
        vis = frame.copy()
        data, points, _ = detector.detectAndDecode(frame)

        if points is None or len(points) == 0:
            results.append({"frame_idx": idx, "found": False, "bbox": None})
            processed.append(vis)
            continue

        pts = points[0].astype(int)
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

        cv2.polylines(vis, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)

        results.append({"frame_idx": idx, "found": True, "bbox": bbox})
        processed.append(vis)

    video_rel = save_as_avi(processed, save_prefix)
    return results, video_rel


# ============================================
# MARKER-LESS TRACKER (clean KLT)
# ============================================
def track_markerless(video_frames, save_prefix="hw5_markerless"):
    if not video_frames:
        return [], None

    processed = []
    results = []

    first = video_frames[0]
    gray_prev = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)

    pts_prev = cv2.goodFeaturesToTrack(
        gray_prev, maxCorners=200, qualityLevel=0.01, minDistance=7
    )

    if pts_prev is None or len(pts_prev) < 5:
        pts_prev = None

    if pts_prev is not None:
        flat = pts_prev.reshape(-1, 2)
        x_min, y_min = flat.min(axis=0)
        x_max, y_max = flat.max(axis=0)
        bbox0 = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    else:
        bbox0 = None

    vis0 = first.copy()
    if bbox0:
        x, y, w, h = bbox0
        cv2.rectangle(
            vis0, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2
        )
    processed.append(vis0)
    results.append({"frame_idx": 0, "found": bbox0 is not None, "bbox": bbox0})

    if pts_prev is None:
        video_rel = save_as_avi(processed, save_prefix)
        return results, video_rel

    for idx in range(1, len(video_frames)):
        frame = video_frames[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, pts_prev, None)

        if next_pts is None:
            pts_prev = None
        else:
            good = next_pts[st.flatten() == 1]
            if len(good) < 5:
                pts_prev = None
            else:
                pts_prev = good.reshape(-1, 1, 2).astype(np.float32)

        if pts_prev is None:
            bbox = None
            found = False
        else:
            flat = pts_prev.reshape(-1, 2)
            x_min, y_min = flat.min(axis=0)
            x_max, y_max = flat.max(axis=0)
            bbox = [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min),
            ]
            found = True

        vis = frame.copy()
        if pts_prev is not None:
            for p in pts_prev.reshape(-1, 2):
                cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(
                vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )

        processed.append(vis)
        results.append({"frame_idx": idx, "found": found, "bbox": bbox})

        gray_prev = gray

    video_rel = save_as_avi(processed, save_prefix)
    return results, video_rel


# ============================================
# SAM2 TRACKER (offline NPZ masks)
# ============================================
def track_sam2_from_npz(video_frames, npz_file, save_prefix="hw5_sam2"):
    if hasattr(npz_file, "read"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        tmp.write(npz_file.read())
        tmp.flush()
        tmp.close()
        path = tmp.name
    else:
        path = npz_file

    data = np.load(path)
    masks = data["masks"]
    if masks.ndim == 4:
        masks = masks[..., 0]

    processed = []
    results = []
    n = min(len(video_frames), masks.shape[0])

    for idx in range(n):
        frame = video_frames[idx]
        mask = (masks[idx] > 0).astype(np.uint8)

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            bbox = None
            found = False
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox = [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min),
            ]
            found = True

        vis = frame.copy()
        if found:
            overlay = vis.copy()
            overlay[mask > 0] = (0, 0, 255)
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)

            x, y, w, h = bbox
            cv2.rectangle(
                vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )

        processed.append(vis)
        results.append({"frame_idx": idx, "has_mask": found, "bbox": bbox})

    video_rel = save_as_avi(processed, save_prefix)
    return results, video_rel


# ============================================
# SUMMARY STATISTICS
# ============================================
def summarize_tracking(results):
    widths = []
    heights = []
    tracked = 0

    for r in results:
        bbox = r.get("bbox")
        if bbox:
            tracked += 1
            widths.append(bbox[2])
            heights.append(bbox[3])

    return {
        "num_frames": len(results),
        "tracked_frames": tracked,
        "coverage": tracked / len(results) if results else 0,
        "mean_width": float(np.mean(widths)) if widths else 0,
        "mean_height": float(np.mean(heights)) if heights else 0,
    }
