import os
import uuid
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np

RESULTS_DIR = os.path.join("static", "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _unique_filename(prefix: str = "hw4") -> str:
    return f"{prefix}_{uuid.uuid4().hex}.png"


# --------------------------
# Common helpers
# --------------------------


def _preprocess_image(image_bgr: np.ndarray, max_dim: int = 700) -> np.ndarray:
    """
    Downscale image to max_dim on the longer side to keep custom SIFT fast.
    """
    h, w = image_bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image_bgr


# --------------------------
# Custom SIFT implementation
# --------------------------


def _gaussian_pyramid(
    gray: np.ndarray, num_scales: int = 3, sigma: float = 1.6
) -> List[np.ndarray]:
    """
    Single-octave Gaussian pyramid: base + num_scales + extra for DoG.
    """
    gray = gray.astype(np.float32) / 255.0
    k = 2 ** (1.0 / num_scales)
    sigmas = [sigma * (k**i) for i in range(num_scales + 3)]
    gaussians = []
    for s in sigmas:
        gaussians.append(cv2.GaussianBlur(gray, (0, 0), s))
    return gaussians


def _dog_pyramid(gaussians: List[np.ndarray]) -> List[np.ndarray]:
    return [gaussians[i + 1] - gaussians[i] for i in range(len(gaussians) - 1)]


def _detect_keypoints(
    dogs: List[np.ndarray],
    contrast_thresh: float = 0.01,
    max_keypoints: int = 400,
) -> List[Dict[str, Any]]:
    """
    Simplified DoG extrema detection.
    Returns at most max_keypoints with largest |DoG| response.
    """
    h, w = dogs[0].shape
    num_scales = len(dogs)
    candidates = []

    for s in range(1, num_scales - 1):
        dog_prev = dogs[s - 1]
        dog_cur = dogs[s]
        dog_next = dogs[s + 1]

        for y in range(1, h - 1):
            row_prev = dog_prev[y - 1 : y + 2]
            row_cur = dog_cur[y - 1 : y + 2]
            row_next = dog_next[y - 1 : y + 2]
            for x in range(1, w - 1):
                val = dog_cur[y, x]
                if abs(val) < contrast_thresh:
                    continue

                patch_prev = row_prev[:, x - 1 : x + 2]
                patch_cur = row_cur[:, x - 1 : x + 2]
                patch_next = row_next[:, x - 1 : x + 2]

                local_stack = np.stack([patch_prev, patch_cur, patch_next], axis=0)
                if val > 0:
                    if val < local_stack.max():
                        continue
                else:
                    if val > local_stack.min():
                        continue

                candidates.append(
                    (abs(val), x, y, s + 1)
                )  # s+1 aligns with Gaussian index

    if not candidates:
        return []

    # keep strongest responses
    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:max_keypoints]

    keypoints = []
    for _, x, y, s_idx in candidates:
        keypoints.append({"x": float(x), "y": float(y), "scale_idx": int(s_idx)})
    return keypoints


def _prepare_gradients(
    gaussians: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Precompute gradient magnitude and orientation for each Gaussian level.
    """
    mags = []
    oris = []
    for img in gaussians:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        ori = cv2.phase(gx, gy, angleInDegrees=True)  # 0..360
        mags.append(mag)
        oris.append(ori)
    return mags, oris


def _assign_orientations(
    mags: List[np.ndarray],
    oris: List[np.ndarray],
    keypoints: List[Dict[str, Any]],
    radius_factor: float = 3.0,
    num_bins: int = 36,
    peak_ratio: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Orientation assignment around each keypoint using histogram of gradient orientations.
    """
    oriented_kps = []

    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        s_idx = kp["scale_idx"]
        mag = mags[s_idx]
        ori = oris[s_idx]

        sigma = 1.6
        radius = int(radius_factor * sigma)
        ix = int(round(x))
        iy = int(round(y))

        if ix < radius or iy < radius:
            continue
        if ix >= mag.shape[1] - radius or iy >= mag.shape[0] - radius:
            continue

        y0 = iy - radius
        y1 = iy + radius + 1
        x0 = ix - radius
        x1 = ix + radius + 1

        patch_mag = mag[y0:y1, x0:x1]
        patch_ori = ori[y0:y1, x0:x1]

        yy, xx = np.mgrid[y0:y1, x0:x1]
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
        weight = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * (sigma**2)))

        hist = np.zeros(num_bins, dtype=np.float32)
        angles = patch_ori
        magnitudes = patch_mag * weight

        bin_f = angles * (num_bins / 360.0)
        bin_idx = np.floor(bin_f).astype(int) % num_bins
        for b, m in zip(bin_idx.ravel(), magnitudes.ravel()):
            hist[b] += m

        max_val = hist.max()
        if max_val <= 0:
            continue

        for bin_id, v in enumerate(hist):
            if v >= peak_ratio * max_val:
                angle = (360.0 * bin_id) / num_bins
                new_kp = dict(kp)
                new_kp["angle"] = angle
                oriented_kps.append(new_kp)

    return oriented_kps


def _compute_descriptors(
    mags: List[np.ndarray],
    oris: List[np.ndarray],
    keypoints: List[Dict[str, Any]],
    descriptor_width: int = 4,
    num_bins: int = 8,
    window_size: int = 4,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    SIFT-like 4x4 cells * 8 bins = 128-D descriptor.
    Uses precomputed magnitude and orientation images.
    """
    descriptors = []
    locations = []

    cell_size = window_size  # pixels per cell side
    half_size = descriptor_width * cell_size // 2  # e.g. 8

    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        s_idx = kp["scale_idx"]
        angle = kp["angle"]
        mag_img = mags[s_idx]
        ori_img = oris[s_idx]

        ix = int(round(x))
        iy = int(round(y))

        if ix - half_size < 1 or iy - half_size < 1:
            continue
        if (
            ix + half_size >= mag_img.shape[1] - 1
            or iy + half_size >= mag_img.shape[0] - 1
        ):
            continue

        cos_t = np.cos(np.deg2rad(angle))
        sin_t = np.sin(np.deg2rad(angle))

        hist = np.zeros(
            (descriptor_width, descriptor_width, num_bins), dtype=np.float32
        )

        for dy in range(-half_size, half_size):
            for dx in range(-half_size, half_size):
                # rotate offset
                rx = cos_t * dx + sin_t * dy
                ry = -sin_t * dx + cos_t * dy

                cx = rx / cell_size + descriptor_width / 2 - 0.5
                cy = ry / cell_size + descriptor_width / 2 - 0.5
                if cx < 0 or cx >= descriptor_width or cy < 0 or cy >= descriptor_width:
                    continue

                px = ix + dx
                py = iy + dy

                m = mag_img[py, px]
                theta = (ori_img[py, px] - angle) % 360.0

                # Gaussian weighting over the descriptor window
                weight = np.exp(
                    -(rx * rx + ry * ry)
                    / (2 * (0.5 * descriptor_width * cell_size) ** 2)
                )
                m *= weight

                bin_o_f = theta * (num_bins / 360.0)
                bin_o = int(np.floor(bin_o_f)) % num_bins

                bx = int(np.floor(cx))
                by = int(np.floor(cy))

                hist[by, bx, bin_o] += m

        desc = hist.flatten()
        norm = np.linalg.norm(desc)
        if norm > 1e-7:
            desc /= norm
            desc = np.clip(desc, 0, 0.2)
            desc /= np.linalg.norm(desc) + 1e-7

        descriptors.append(desc.astype(np.float32))
        locations.append((x, y))

    if not descriptors:
        return np.zeros(
            (0, descriptor_width * descriptor_width * num_bins), dtype=np.float32
        ), []

    return np.vstack(descriptors), locations


def _sift_features_custom(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Full custom SIFT-like pipeline: Gaussian pyramid, DoG, keypoints, orientation, descriptor.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gaussians = _gaussian_pyramid(gray, num_scales=3, sigma=1.6)
    dogs = _dog_pyramid(gaussians)
    keypoints = _detect_keypoints(dogs, contrast_thresh=0.01, max_keypoints=400)
    if not keypoints:
        return np.zeros((0, 128), dtype=np.float32), []

    mags, oris = _prepare_gradients(gaussians)
    oriented_kps = _assign_orientations(mags, oris, keypoints)
    if not oriented_kps:
        return np.zeros((0, 128), dtype=np.float32), []

    descriptors, locations = _compute_descriptors(mags, oris, oriented_kps)
    return descriptors, locations


def _match_descriptors(
    desc1: np.ndarray, desc2: np.ndarray, ratio: float = 0.75
) -> List[Tuple[int, int]]:
    """
    Brute-force L2 matching with Lowe's ratio test.
    """
    matches = []
    if desc1.shape[0] == 0 or desc2.shape[0] == 0:
        return matches

    for i in range(desc1.shape[0]):
        d = desc1[i]
        diff = desc2 - d
        dist = np.linalg.norm(diff, axis=1)
        if dist.shape[0] < 2:
            continue
        idxs = np.argsort(dist)
        d1 = dist[idxs[0]]
        d2 = dist[idxs[1]]
        if d1 < ratio * d2:
            matches.append((i, int(idxs[0])))
    return matches


def _compute_homography(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """
    DLT homography: dst ~ H * src.
    """
    n = src_pts.shape[0]
    A = []
    for i in range(n):
        x, y = src_pts[i]
        u, v = dst_pts[i]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    H /= H[2, 2] + 1e-12
    return H


def _ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_iterations: int = 400,
    reproj_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manual RANSAC for homography.
    """
    num_points = src_pts.shape[0]
    if num_points < 4:
        return None, None

    best_H = None
    best_inliers = None
    best_count = 0

    src_h = np.hstack([src_pts, np.ones((num_points, 1))])

    for _ in range(num_iterations):
        idxs = np.random.choice(num_points, 4, replace=False)
        sp = src_pts[idxs]
        dp = dst_pts[idxs]

        try:
            H = _compute_homography(sp, dp)
        except np.linalg.LinAlgError:
            continue

        proj = (H @ src_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        errors = np.linalg.norm(proj - dst_pts, axis=1)
        inliers = errors < reproj_thresh
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None:
        return None, None

    if np.sum(best_inliers) >= 4:
        H_refined = _compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
    else:
        H_refined = best_H

    return H_refined, best_inliers


def _pairwise_homography_custom(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Homography from img2 to img1 using custom SIFT + custom RANSAC.
    """
    desc1, loc1 = _sift_features_custom(img1)
    desc2, loc2 = _sift_features_custom(img2)

    matches = _match_descriptors(desc1, desc2)
    if len(matches) < 4:
        return None, {
            "num_matches": len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    src_pts = np.float32([loc2[j] for (_, j) in matches])
    dst_pts = np.float32([loc1[i] for (i, _) in matches])

    H, inliers = _ransac_homography(src_pts, dst_pts)
    if H is None or inliers is None:
        return None, {
            "num_matches": len(matches),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    num_inliers = int(np.sum(inliers))
    inlier_ratio = float(num_inliers) / max(len(matches), 1)

    stats = {
        "num_matches": len(matches),
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
    }
    return H, stats


# --------------------------
# OpenCV SIFT pipeline
# --------------------------


def _pairwise_homography_opencv(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Homography from img2 to img1 using OpenCV SIFT + findHomography(RANSAC).
    """
    # Use same downscaling for fairness and speed
    img1 = _preprocess_image(img1)
    img2 = _preprocess_image(img2)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.xfeatures2d.SIFT_create()

    kps1, desc1 = sift.detectAndCompute(gray1, None)
    kps2, desc2 = sift.detectAndCompute(gray2, None)

    if desc1 is None or desc2 is None or len(kps1) == 0 or len(kps2) == 0:
        return None, {
            "num_matches": 0,
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for m, n in raw_matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 4:
        return None, {
            "num_matches": len(good),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    src_pts = np.float32([kps2[m.trainIdx].pt for m in good])
    dst_pts = np.float32([kps1[m.queryIdx].pt for m in good])

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return None, {
            "num_matches": len(good),
            "num_inliers": 0,
            "inlier_ratio": 0.0,
        }

    num_inliers = int(mask.ravel().sum())
    inlier_ratio = float(num_inliers) / max(len(good), 1)

    stats = {
        "num_matches": len(good),
        "num_inliers": num_inliers,
        "inlier_ratio": inlier_ratio,
    }
    return H, stats


# --------------------------
# Stitching and blending
# --------------------------


def _warp_and_blend(
    images: List[np.ndarray], homographies: List[np.ndarray], prefix: str
) -> str:
    """
    Warp all images into a common canvas and blend with feathered weights.
    Uses the middle image as the reference view to reduce distortion.
    """
    _ensure_results_dir()

    num_images = len(images)
    if num_images == 0:
        raise ValueError("No images to stitch.")

    # --- Re-center homographies around the middle image to reduce distortion ---
    # Incoming homographies map each image -> image 0.
    mid = num_images // 2
    H_mid = homographies[mid].astype(np.float64)
    H_mid_inv = np.linalg.inv(H_mid + 1e-12 * np.eye(3))

    centered_H = []
    for H in homographies:
        Hc = H_mid_inv @ H  # map each image into "middle" frame coordinates
        Hc /= Hc[2, 2] + 1e-12
        centered_H.append(Hc.astype(np.float32))

    # --- Compute canvas bounds with padding ---
    corners_all = []
    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        warped = (centered_H[i] @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        corners_all.append(warped)

    corners_all = np.vstack(corners_all)
    pad = 100  # pixels of padding to avoid tight clipping
    x_min = int(np.floor(corners_all[:, 0].min())) - pad
    x_max = int(np.ceil(corners_all[:, 0].max())) + pad
    y_min = int(np.floor(corners_all[:, 1].min())) - pad
    y_max = int(np.ceil(corners_all[:, 1].max())) + pad

    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    width = x_max - x_min
    height = y_max - y_min

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    # --- Allocate accumulators ---
    acc = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    # --- Warp each image with a feathered mask ---
    for i, img in enumerate(images):
        img_f = img.astype(np.float32)

        h, w = img.shape[:2]
        # mask = 1 inside the image, 0 outside, then distance transform for feathering
        mask = np.ones((h, w), dtype=np.uint8) * 255
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        if dist.max() > 0:
            dist = dist / dist.max()
        else:
            dist = np.ones_like(dist, dtype=np.float32)

        H_total = T @ centered_H[i]
        warped_img = cv2.warpPerspective(img_f, H_total, (width, height))
        warped_w = cv2.warpPerspective(dist, H_total, (width, height))

        warped_w = warped_w[..., None].astype(np.float32)

        acc += warped_img * warped_w
        weight += warped_w

    # Avoid division by zero
    weight[weight == 0] = 1.0
    pano = acc / weight
    pano = np.clip(pano, 0, 255).astype(np.uint8)

    filename = _unique_filename(prefix)
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, pano)
    return filename


# --------------------------
# Public API
# --------------------------


def stitch_images_custom(images: List[np.ndarray]) -> Dict[str, Any]:
    """
    Custom SIFT + custom RANSAC stitching.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    # Preprocess (downscale) all images
    proc_images = [_preprocess_image(img) for img in images]

    homographies = [np.eye(3, dtype=np.float32)]
    pair_stats = []

    for i in range(1, len(proc_images)):
        H, stats = _pairwise_homography_custom(proc_images[i - 1], proc_images[i])
        stats["pair"] = f"{i - 1}-{i}"
        pair_stats.append(stats)

        if H is None:
            raise ValueError(
                f"Custom SIFT failed to compute homography for pair {i - 1}-{i}."
            )

        H_to_base = homographies[i - 1] @ H
        homographies.append(H_to_base.astype(np.float32))

    filename = _warp_and_blend(proc_images, homographies, prefix="hw4_custom")

    return {
        "filename": filename,
        "stats": {
            "num_images": len(proc_images),
            "pair_stats": pair_stats,
        },
    }


def stitch_images_opencv(images: List[np.ndarray]) -> Dict[str, Any]:
    """
    OpenCV SIFT + OpenCV RANSAC stitching.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    # Use preprocessed images for consistency
    proc_images = [_preprocess_image(img) for img in images]

    homographies = [np.eye(3, dtype=np.float32)]
    pair_stats = []

    for i in range(1, len(proc_images)):
        H, stats = _pairwise_homography_opencv(proc_images[i - 1], proc_images[i])
        stats["pair"] = f"{i - 1}-{i}"
        pair_stats.append(stats)

        if H is None:
            raise ValueError(
                f"OpenCV SIFT failed to compute homography for pair {i - 1}-{i}."
            )

        H_to_base = homographies[i - 1] @ H
        homographies.append(H_to_base.astype(np.float32))

    filename = _warp_and_blend(proc_images, homographies, prefix="hw4_opencv")

    return {
        "filename": filename,
        "stats": {
            "num_images": len(proc_images),
            "pair_stats": pair_stats,
        },
    }


def store_mobile_panorama(image: np.ndarray) -> Dict[str, Any]:
    """
    Save a single mobile panorama image to static/results/ and return filename.
    """
    _ensure_results_dir()
    filename = _unique_filename("hw4_mobile")
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, image)
    return {"filename": filename}
