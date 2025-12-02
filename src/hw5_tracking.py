import os
import tempfile

import cv2
import numpy as np

# Ensure results directory exists
RESULTS_DIR = os.path.join("static", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def decode_video_file(file_storage_or_path, max_frames=None):
    """
    Decode an uploaded video (Flask FileStorage) or a file path into a list of frames (BGR NumPy arrays).
    """
    if hasattr(file_storage_or_path, "read"):
        # Handle Flask/Werkzeug FileStorage-like objects
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        data = file_storage_or_path.read()
        tmp.write(data)
        tmp.flush()
        tmp.close()
        path = tmp.name
    else:
        path = file_storage_or_path

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not open video file for decoding.")

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()
    return frames


def save_frame_sequence(frames, prefix):
    """
    Save a sequence of frames into static/results with a prefix.
    Returns a list of relative filenames, e.g. 'results/prefix_0000.jpg'.
    """
    filenames = []
    for idx, frame in enumerate(frames):
        rel_name = os.path.join("results", f"{prefix}_{idx:04d}.jpg")
        full_path = os.path.join("static", rel_name)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        cv2.imwrite(full_path, frame)
        filenames.append(rel_name)
    return filenames


def _detect_aruco_in_frame(frame, dictionary_name="DICT_4X4_50"):
    """
    Detect a single ArUco marker in the frame.
    Returns (corners, bbox) or (None, None).
    bbox = [x, y, w, h]
    """
    if not hasattr(cv2, "aruco"):
        return None, None

    aruco = cv2.aruco
    dict_id = getattr(aruco, dictionary_name, aruco.DICT_4X4_50)
    dictionary = aruco.Dictionary_get(dict_id)
    parameters = aruco.DetectorParameters_create()

    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
    if ids is None or len(corners) == 0:
        return None, None

    c = corners[0].reshape(-1, 2)
    x_min, y_min = np.min(c, axis=0)
    x_max, y_max = np.max(c, axis=0)
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    return c.astype(int), bbox


def _detect_qr_in_frame(frame):
    """
    Detect a QR code in the frame.
    Returns (corners, bbox) or (None, None).
    bbox = [x, y, w, h]
    """
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(frame)
    if points is None or len(points) == 0:
        return None, None

    pts = points[0]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    return pts.astype(int), bbox


def track_marker(video_frames, marker_type="aruco", save_prefix="hw5_marker"):
    """
    Marker-based tracking using ArUco or QR markers.

    Args:
        video_frames: list of BGR frames (NumPy arrays).
        marker_type: 'aruco' or 'qr'.
        save_prefix: prefix for saved overlay frames.

    Returns:
        results: list of dicts with keys {frame_idx, found, bbox}
        filenames: list of saved frame filenames under static/results
    """
    processed_frames = []
    results = []

    if marker_type not in ("aruco", "qr"):
        marker_type = "aruco"

    for idx, frame in enumerate(video_frames):
        frame_vis = frame.copy()
        corners = None
        bbox = None

        if marker_type == "aruco" and hasattr(cv2, "aruco"):
            corners, bbox = _detect_aruco_in_frame(frame_vis)
        elif marker_type == "qr" or not hasattr(cv2, "aruco"):
            corners, bbox = _detect_qr_in_frame(frame_vis)

        found = bbox is not None

        if found:
            x, y, w, h = bbox
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv2.rectangle(frame_vis, pt1, pt2, (0, 255, 0), 2)

            if corners is not None:
                cv2.polylines(
                    frame_vis, [corners.reshape(-1, 1, 2)], True, (255, 0, 0), 2
                )

        results.append(
            {
                "frame_idx": int(idx),
                "found": bool(found),
                "bbox": bbox,
            }
        )
        processed_frames.append(frame_vis)

    filenames = save_frame_sequence(processed_frames, save_prefix)
    return results, filenames


def _init_klt_points(frame, init_bbox=None, max_corners=100):
    """
    Initialize KLT points inside init_bbox or across the whole frame if init_bbox is None.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = None
    if init_bbox is not None:
        x, y, w, h = [int(v) for v in init_bbox]
        mask = np.zeros_like(gray)
        mask[y : y + h, x : x + w] = 255

    pts = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=7,
        mask=mask,
    )
    return gray, pts


def _bbox_from_points(pts):
    """
    Compute [x, y, w, h] bounding box from a set of 2D points.
    """
    if pts is None or len(pts) == 0:
        return None
    pts_reshaped = pts.reshape(-1, 2)
    x_min, y_min = np.min(pts_reshaped, axis=0)
    x_max, y_max = np.max(pts_reshaped, axis=0)
    return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


def track_markerless(video_frames, init_bbox=None, save_prefix="hw5_markerless"):
    """
    Marker-less tracking using KLT optical flow.

    Args:
        video_frames: list of BGR frames (NumPy arrays).
        init_bbox: optional [x, y, w, h] on the first frame; if None, auto-initialized.
        save_prefix: prefix for saved overlay frames.

    Returns:
        results: list of dicts with keys {frame_idx, found, bbox}
        filenames: list of saved frame filenames under static/results
    """
    if not video_frames:
        return [], []

    first_frame = video_frames[0]
    prev_gray, prev_pts = _init_klt_points(first_frame, init_bbox)
    init_bbox_auto = _bbox_from_points(prev_pts) if init_bbox is None else init_bbox

    processed_frames = []
    results = []

    # Process first frame
    bbox0 = init_bbox_auto
    frame_vis0 = first_frame.copy()
    if bbox0 is not None:
        x, y, w, h = bbox0
        cv2.rectangle(
            frame_vis0, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 255), 2
        )
    results.append(
        {
            "frame_idx": 0,
            "found": bbox0 is not None,
            "bbox": bbox0,
        }
    )
    processed_frames.append(frame_vis0)

    if prev_pts is None:
        filenames = save_frame_sequence(processed_frames, save_prefix)
        return results, filenames

    # Track through remaining frames
    prev_pts_valid = prev_pts
    prev_gray_valid = prev_gray

    for idx in range(1, len(video_frames)):
        frame = video_frames[idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray_valid, gray, prev_pts_valid, None
        )

        if next_pts is None or status is None:
            bbox = None
            found = False
            pts_to_draw = None
        else:
            good_new = next_pts[status.flatten() == 1]
            if len(good_new) == 0:
                bbox = None
                found = False
                pts_to_draw = None
            else:
                bbox = _bbox_from_points(good_new)
                found = bbox is not None
                pts_to_draw = good_new

                prev_pts_valid = good_new.reshape(-1, 1, 2)
                prev_gray_valid = gray

        frame_vis = frame.copy()
        if pts_to_draw is not None:
            for p in pts_to_draw:
                cv2.circle(frame_vis, (int(p[0]), int(p[1])), 2, (0, 0, 255), -1)

        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(
                frame_vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2
            )

        results.append(
            {
                "frame_idx": int(idx),
                "found": bool(found),
                "bbox": bbox,
            }
        )
        processed_frames.append(frame_vis)

    filenames = save_frame_sequence(processed_frames, save_prefix)
    return results, filenames


def track_sam2_from_npz(video_frames, npz_source, save_prefix="hw5_sam2"):
    """
    Tracking / segmentation visualization based on precomputed SAM2 masks stored in an NPZ file.

    Args:
        video_frames: list of BGR frames (NumPy arrays).
        npz_source: path string or FileStorage-like object with .read().
        save_prefix: prefix for saved overlay frames.

    NPZ format expectation:
        masks: array of shape (T, H, W) or (T, H, W, 1), binary or labeled.

    Returns:
        results: list of dicts with keys {frame_idx, has_mask, bbox}
        filenames: list of saved frame filenames under static/results
    """
    # Allow passing a FileStorage-like object directly
    if hasattr(npz_source, "read"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".npz")
        data_bytes = npz_source.read()
        tmp.write(data_bytes)
        tmp.flush()
        tmp.close()
        npz_path = tmp.name
    else:
        npz_path = npz_source

    data = np.load(npz_path)
    if "masks" not in data:
        raise ValueError("NPZ file must contain 'masks' array.")

    masks = data["masks"]
    if masks.ndim == 4 and masks.shape[-1] == 1:
        masks = masks[..., 0]

    n_frames = min(len(video_frames), masks.shape[0])

    processed_frames = []
    results = []

    for idx in range(n_frames):
        frame = video_frames[idx]
        mask = masks[idx]

        # Binarize mask
        if mask.dtype != np.uint8:
            mask_bin = (mask > 0).astype(np.uint8)
        else:
            mask_bin = (mask > 0).astype(np.uint8)

        ys, xs = np.where(mask_bin > 0)
        if len(xs) == 0 or len(ys) == 0:
            bbox = None
            has_mask = False
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()
            bbox = [
                float(x_min),
                float(y_min),
                float(x_max - x_min),
                float(y_max - y_min),
            ]
            has_mask = True

        frame_vis = frame.copy()

        # Overlay mask with simple alpha blending
        if has_mask:
            overlay = frame_vis.copy()
            overlay[mask_bin > 0] = (0, 0, 255)
            alpha = 0.4
            frame_vis = cv2.addWeighted(overlay, alpha, frame_vis, 1 - alpha, 0)

            x, y, w, h = bbox
            cv2.rectangle(
                frame_vis,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                (0, 255, 0),
                2,
            )

        results.append(
            {
                "frame_idx": int(idx),
                "has_mask": bool(has_mask),
                "bbox": bbox,
            }
        )
        processed_frames.append(frame_vis)

    filenames = save_frame_sequence(processed_frames, save_prefix)
    return results, filenames


def summarize_tracking(results):
    """
    Compute simple tracking statistics from a list of per-frame result dicts.

    Each result dict is expected to contain:
        - 'bbox': [x, y, w, h] or None
        - 'frame_idx': int

    Returns:
        dict with aggregate statistics.
    """
    num_frames = len(results)
    num_tracked = 0
    widths = []
    heights = []

    for r in results:
        bbox = r.get("bbox")
        if bbox is not None:
            num_tracked += 1
            widths.append(bbox[2])
            heights.append(bbox[3])

    mean_w = float(np.mean(widths)) if widths else 0.0
    mean_h = float(np.mean(heights)) if heights else 0.0
    coverage = float(num_tracked) / num_frames if num_frames > 0 else 0.0

    return {
        "num_frames": int(num_frames),
        "num_tracked": int(num_tracked),
        "tracking_coverage": coverage,
        "mean_bbox_width": mean_w,
        "mean_bbox_height": mean_h,
    }
