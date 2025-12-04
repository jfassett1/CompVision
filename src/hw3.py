import os
import math
import uuid
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

RESULTS_DIR = os.path.join("static", "results")


def _ensure_results_dir() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_image(image: np.ndarray, prefix: str, ext: str = ".png") -> str:
    _ensure_results_dir()
    unique_id = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{unique_id}{ext}"
    filepath = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(filepath, image)
    # Return path relative to static/ so app.py can use url_for("static", filename=path)
    return os.path.join("results", filename)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 1. GRADIENTS + LoG -----------------------------------------------------------


def compute_gradients_and_log(images: List[np.ndarray]) -> List[Dict[str, Any]]:
    """
    For each input BGR image:
      - Convert to grayscale.
      - Compute Sobel gradients Gx, Gy.
      - Compute magnitude and angle.
      - Normalize and save magnitude and angle visualizations.
      - Compute Laplacian of Gaussian and save.
    Returns list of dicts with:
      grad_mag_path, grad_angle_path, log_path, mean_grad_mag
    Paths are relative to static/.
    """
    results: List[Dict[str, Any]] = []
    _ensure_results_dir()

    for idx, img in enumerate(images):
        gray = _to_gray(img)

        # Gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        mag = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=True)  # 0–360 degrees

        mean_grad_mag = float(np.mean(mag))

        # Normalize magnitude to 0–255
        mag_norm = np.zeros_like(mag)
        cv2.normalize(mag, mag_norm, 0, 255, cv2.NORM_MINMAX)
        mag_uint8 = mag_norm.astype(np.uint8)

        # Map angle (0–360) to 0–255 for grayscale visualization
        angle_norm = (angle / 360.0) * 255.0
        angle_uint8 = angle_norm.astype(np.uint8)

        grad_mag_path = _save_image(mag_uint8, f"hw3_grad_mag_{idx}")
        grad_angle_path = _save_image(angle_uint8, f"hw3_grad_angle_{idx}")

        # Laplacian of Gaussian
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        log = cv2.Laplacian(blurred, cv2.CV_32F, ksize=3)
        log_norm = np.zeros_like(log)
        cv2.normalize(log, log_norm, 0, 255, cv2.NORM_MINMAX)
        log_uint8 = np.abs(log_norm).astype(np.uint8)

        log_path = _save_image(log_uint8, f"hw3_log_{idx}")

        results.append(
            {
                "grad_mag_path": grad_mag_path,
                "grad_angle_path": grad_angle_path,
                "log_path": log_path,
                "mean_grad_mag": mean_grad_mag,
                "index": idx,
            }
        )

    return results


# 2. EDGE AND CORNER KEYPOINT DETECTORS ---------------------------------------


def detect_edge_keypoints(
    image: np.ndarray, grad_mag: np.ndarray = None
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Simple edge keypoint detector:
      - Uses gradient magnitude (Sobel) if not provided.
      - Threshold = mean + 0.5 * std.
      - Non-maximum suppression via 3x3 local maxima.
      - Keypoints are local maxima above threshold.
    Returns:
      - list of (x, y) keypoints
      - overlay image with keypoints drawn on top of original BGR image
    """
    if image.ndim == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if grad_mag is None:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)

    mean_val = float(np.mean(grad_mag))
    std_val = float(np.std(grad_mag))
    threshold = mean_val + 0.5 * std_val

    # Local maxima mask
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(grad_mag, kernel)
    local_max_mask = (grad_mag == dilated) & (grad_mag > threshold)

    ys, xs = np.where(local_max_mask)
    strengths = grad_mag[ys, xs]

    # Keep strongest up to a max count to avoid clutter
    max_points = 500
    if len(strengths) > max_points:
        idx_sorted = np.argsort(strengths)[-max_points:]
        xs = xs[idx_sorted]
        ys = ys[idx_sorted]

    keypoints = [(int(x), int(y)) for x, y in zip(xs, ys)]

    for x, y in keypoints:
        cv2.circle(color, (x, y), 2, (0, 0, 255), -1)

    return keypoints, color


def detect_corner_keypoints(
    image: np.ndarray,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """
    Simple Harris-like corner detector:
      - Compute Ix, Iy with Sobel.
      - Compute second moment matrix components, smoothed with Gaussian.
      - R = det(M) - k * (trace(M)^2).
      - Threshold and non-maximum suppression.
    Returns:
      - list of (x, y) keypoints
      - overlay image with keypoints drawn.
    """
    if image.ndim == 2:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_f = gray.astype(np.float32)

    Ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    # Smooth second moment components
    Sxx = cv2.GaussianBlur(Ixx, (5, 5), 1.0)
    Syy = cv2.GaussianBlur(Iyy, (5, 5), 1.0)
    Sxy = cv2.GaussianBlur(Ixy, (5, 5), 1.0)

    k = 0.04
    detM = Sxx * Syy - Sxy * Sxy
    traceM = Sxx + Syy
    R = detM - k * (traceM**2)

    # Normalize R for robust thresholding
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    thresh = float(0.01)  # heuristic threshold
    corner_mask = R_norm > thresh

    # Non-maximum suppression
    R_uint = (R_norm * 255).astype(np.uint8)
    dilated = cv2.dilate(R_uint, None)
    nms_mask = (R_uint == dilated) & corner_mask

    ys, xs = np.where(nms_mask)
    strengths = R_norm[ys, xs]

    max_points = 500
    if len(strengths) > max_points:
        idx_sorted = np.argsort(strengths)[-max_points:]
        xs = xs[idx_sorted]
        ys = ys[idx_sorted]

    keypoints = [(int(x), int(y)) for x, y in zip(xs, ys)]

    for x, y in keypoints:
        cv2.circle(color, (x, y), 3, (0, 255, 0), 1)

    return keypoints, color


def run_edge_corner_detection(image: np.ndarray) -> Dict[str, Any]:
    """
    Wrapper:
      - Computes gradient magnitude for edge detection.
      - Runs edge keypoint detection.
      - Runs corner keypoint detection.
      - Saves overlay images.
    Returns dict with:
      edge_overlay_path, corner_overlay_path, edge_count, corner_count
    """
    gray = _to_gray(image)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)

    edge_points, edge_overlay = detect_edge_keypoints(image, grad_mag=grad_mag)
    corner_points, corner_overlay = detect_corner_keypoints(image)

    edge_overlay_path = _save_image(edge_overlay, "hw3_edge_overlay")
    corner_overlay_path = _save_image(corner_overlay, "hw3_corner_overlay")

    return {
        "edge_overlay_path": edge_overlay_path,
        "corner_overlay_path": corner_overlay_path,
        "edge_count": len(edge_points),
        "corner_count": len(corner_points),
    }


# 3. EXACT OBJECT BOUNDARY EXTRACTION -----------------------------------------


def find_object_boundary(image: np.ndarray) -> Dict[str, Any]:
    """
    Find the boundary of the main object in the image:
      - Convert to grayscale, blur.
      - Otsu threshold.
      - Morphological closing to clean mask.
      - Find external contours and choose the largest.
      - Draw boundary overlay and mask.
    Returns:
      boundary_overlay_path, mask_path, area, perimeter, bbox
    """
    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = image
    else:
        color = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Try both foreground/background to be robust
    # Choose the one with larger main contour
    candidates = [thresh, cv2.bitwise_not(thresh)]
    best_contour = None
    best_area = 0
    best_mask = None

    for mask in candidates:
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > best_area:
            best_area = area
            best_contour = largest
            best_mask = closed

    if best_contour is None:
        # Fallback: no contour found
        overlay_path = _save_image(color, "hw3_boundary_overlay_none")
        return {
            "boundary_overlay_path": overlay_path,
            "mask_path": None,
            "area": 0.0,
            "perimeter": 0.0,
            "bbox": None,
        }

    perimeter = cv2.arcLength(best_contour, True)
    x, y, w, h = cv2.boundingRect(best_contour)
    bbox = [int(x), int(y), int(w), int(h)]

    overlay = color.copy()
    cv2.drawContours(overlay, [best_contour], -1, (0, 0, 255), 2)

    mask_img = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask_img, [best_contour], -1, 255, thickness=-1)

    boundary_overlay_path = _save_image(overlay, "hw3_boundary_overlay")
    mask_path = _save_image(mask_img, "hw3_boundary_mask")

    return {
        "boundary_overlay_path": boundary_overlay_path,
        "mask_path": mask_path,
        "area": float(best_area),
        "perimeter": float(perimeter),
        "bbox": bbox,
    }


# 4. NON-RECTANGULAR OBJECT SEGMENTATION WITH ARUCO ---------------------------


def _detect_aruco_markers(gray: np.ndarray):
    """
    Helper to detect ArUco markers.
    Returns (corners, ids).
    """
    # Default to a common small dictionary; change if your markers differ
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def segment_object_with_aruco(images: List[np.ndarray]) -> Dict[str, Any]:
    """
    For each image:
      - Detect ArUco markers.
      - Use all detected marker corners to approximate the object boundary
        via their convex hull.
      - Draw markers and hull on overlay, save overlay.
    Returns:
      overlay_paths: list[str]
      summary: {
        "markers_per_image": [...],
        "hull_points_per_image": [...]
      }
    """
    overlay_paths: List[str] = []
    markers_per_image: List[int] = []
    hull_points_per_image: List[int] = []

    for idx, img in enumerate(images):
        if img.ndim == 2:
            gray = img
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            color = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        try:
            corners, ids = _detect_aruco_markers(gray)
        except Exception:
            # If detector fails, just pass through image
            corners, ids = [], None

        if corners is None:
            corners = []

        markers_per_image.append(len(corners))

        overlay = color.copy()

        if corners:
            cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

            # Collect all marker corner points
            pts = []
            for c in corners:
                # c shape: (1, 4, 2)
                pts.append(c.reshape(-1, 2))
            pts_all = np.vstack(pts).astype(np.float32)

            if len(pts_all) >= 3:
                hull = cv2.convexHull(pts_all)
                hull_points_per_image.append(int(len(hull)))
                hull_int = hull.astype(np.int32)
                cv2.polylines(
                    overlay, [hull_int], isClosed=True, color=(0, 255, 0), thickness=2
                )
            else:
                hull_points_per_image.append(0)
        else:
            hull_points_per_image.append(0)

        overlay_path = _save_image(overlay, f"hw3_aruco_overlay_{idx}")
        overlay_paths.append(overlay_path)

    summary = {
        "markers_per_image": markers_per_image,
        "hull_points_per_image": hull_points_per_image,
        "num_images": len(images),
    }

    return {
        "overlay_paths": overlay_paths,
        "summary": summary,
    }


def save_segmented_sequence(images: List[np.ndarray], prefix: str = "hw3_aruco"):
    """
    Convenience wrapper if needed elsewhere. Not directly used by app.py.
    """
    result = segment_object_with_aruco(images)
    return result
