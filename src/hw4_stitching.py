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
# Custom SIFT implementation
# --------------------------


def _gaussian_pyramid(
    gray: np.ndarray, num_scales: int = 5, sigma: float = 1.6
) -> List[np.ndarray]:
    """
    Build a simple Gaussian pyramid (single octave, multiple scales).
    """
    gray = gray.astype(np.float32) / 255.0
    k = 2 ** (1.0 / num_scales)
    sigmas = [sigma * (k**i) for i in range(num_scales + 3)]  # extra for DoG as in SIFT
    gaussians = []
    for s in sigmas:
        gaussians.append(cv2.GaussianBlur(gray, ksize=(0, 0), sigmaX=s, sigmaY=s))
    return gaussians


def _dog_pyramid(gaussians: List[np.ndarray]) -> List[np.ndarray]:
    dogs = []
    for i in range(1, len(gaussians)):
        dogs.append(gaussians[i] - gaussians[i - 1])
    return dogs


def _detect_keypoints(
    gaussians: List[np.ndarray],
    dogs: List[np.ndarray],
    contrast_thresh: float = 0.03,
    edge_thresh: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Very simplified SIFT-like keypoint detection in DoG scale space.
    """
    keypoints = []
    num_scales = len(dogs)
    h, w = dogs[0].shape

    for s in range(1, num_scales - 1):
        dog_prev = dogs[s - 1]
        dog_cur = dogs[s]
        dog_next = dogs[s + 1]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                val = dog_cur[y, x]
                if abs(val) < contrast_thresh:
                    continue

                patch_prev = dog_prev[y - 1 : y + 2, x - 1 : x + 2]
                patch_cur = dog_cur[y - 1 : y + 2, x - 1 : x + 2]
                patch_next = dog_next[y - 1 : y + 2, x - 1 : x + 2]

                local_patch = np.stack([patch_prev, patch_cur, patch_next], axis=0)
                if val > 0:
                    if val < local_patch.max():
                        continue
                else:
                    if val > local_patch.min():
                        continue

                # Edge response rejection using Hessian at scale s
                img = gaussians[s + 1]  # roughly aligned with dog_cur
                dx = (img[y, x + 1] - img[y, x - 1]) * 0.5
                dy = (img[y + 1, x] - img[y - 1, x]) * 0.5
                dxx = img[y, x + 1] - 2 * img[y, x] + img[y, x - 1]
                dyy = img[y + 1, x] - 2 * img[y, x] + img[y - 1, x]
                dxy = (
                    img[y + 1, x + 1]
                    - img[y + 1, x - 1]
                    - img[y - 1, x + 1]
                    + img[y - 1, x - 1]
                ) * 0.25

                tr = dxx + dyy
                det = dxx * dyy - dxy * dxy
                if det <= 0:
                    continue
                r = edge_thresh
                if (tr * tr) / det >= ((r + 1) ** 2) / r:
                    continue

                keypoints.append(
                    {
                        "x": float(x),
                        "y": float(y),
                        "scale_idx": s + 1,  # align with gaussians index
                    }
                )

    return keypoints


def _assign_orientations(
    gaussians: List[np.ndarray],
    keypoints: List[Dict[str, Any]],
    radius_factor: float = 3.0,
    num_bins: int = 36,
    peak_ratio: float = 0.8,
) -> List[Dict[str, Any]]:
    """
    Assign dominant orientation(s) to each keypoint.
    """
    oriented_kps = []

    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        s_idx = kp["scale_idx"]
        img = gaussians[s_idx]

        sigma = 1.6  # approximate
        radius = int(radius_factor * sigma)
        hist = np.zeros(num_bins, dtype=np.float32)

        y0 = max(1, int(round(y)) - radius)
        y1 = min(img.shape[0] - 2, int(round(y)) + radius)
        x0 = max(1, int(round(x)) - radius)
        x1 = min(img.shape[1] - 2, int(round(x)) + radius)

        for yy in range(y0, y1 + 1):
            for xx in range(x0, x1 + 1):
                dx = img[yy, xx + 1] - img[yy, xx - 1]
                dy = img[yy - 1, xx] - img[yy + 1, xx]
                mag = np.sqrt(dx * dx + dy * dy)
                angle = np.degrees(np.arctan2(dy, dx)) % 360.0

                weight = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma * sigma))
                bin_idx = int(round(angle * num_bins / 360.0)) % num_bins
                hist[bin_idx] += weight * mag

        max_val = hist.max()
        if max_val <= 0:
            continue

        for bin_idx, val in enumerate(hist):
            if val >= peak_ratio * max_val:
                angle = (360.0 * bin_idx) / num_bins
                kp_copy = dict(kp)
                kp_copy["angle"] = angle
                oriented_kps.append(kp_copy)

    return oriented_kps


def _compute_descriptors(
    gaussians: List[np.ndarray],
    keypoints: List[Dict[str, Any]],
    window_width: int = 4,
    num_bins: int = 8,
    descriptor_width: int = 4,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Compute SIFT-like 128D descriptors.
    """
    descriptors = []
    locations = []

    for kp in keypoints:
        x = kp["x"]
        y = kp["y"]
        s_idx = kp["scale_idx"]
        angle = np.deg2rad(kp["angle"])
        img = gaussians[s_idx]

        # 16x16 window around keypoint
        half_width = window_width * descriptor_width // 2  # 8 for 4x4
        ix = int(round(x))
        iy = int(round(y))

        if ix - half_width < 1 or ix + half_width >= img.shape[1] - 1:
            continue
        if iy - half_width < 1 or iy + half_width >= img.shape[0] - 1:
            continue

        cos_t = np.cos(angle)
        sin_t = np.sin(angle)
        hist_tensor = np.zeros(
            (descriptor_width, descriptor_width, num_bins), dtype=np.float32
        )

        for dy in range(-half_width, half_width):
            for dx in range(-half_width, half_width):
                rx = cos_t * dx + sin_t * dy
                ry = -sin_t * dx + cos_t * dy

                bin_x = (rx / window_width) + descriptor_width / 2 - 0.5
                bin_y = (ry / window_width) + descriptor_width / 2 - 0.5
                if (
                    bin_x < 0
                    or bin_x >= descriptor_width
                    or bin_y < 0
                    or bin_y >= descriptor_width
                ):
                    continue

                xp = ix + dx
                yp = iy + dy

                gx = img[yp, xp + 1] - img[yp, xp - 1]
                gy = img[yp - 1, xp] - img[yp + 1, xp]
                mag = np.sqrt(gx * gx + gy * gy)
                theta = (np.arctan2(gy, gx) - angle) % (2 * np.pi)
                bin_o = theta * num_bins / (2 * np.pi)

                mag_weight = mag * np.exp(
                    -(rx * rx + ry * ry)
                    / (2 * (0.5 * descriptor_width * window_width) ** 2)
                )

                bx0 = int(np.floor(bin_x))
                by0 = int(np.floor(bin_y))
                bo0 = int(np.floor(bin_o))

                hist_tensor[by0, bx0, bo0 % num_bins] += mag_weight

        desc = hist_tensor.flatten()
        norm = np.linalg.norm(desc)
        if norm > 1e-7:
            desc = desc / norm
            desc = np.clip(desc, 0, 0.2)
            desc = desc / (np.linalg.norm(desc) + 1e-7)

        descriptors.append(desc.astype(np.float32))
        locations.append((x, y))

    if len(descriptors) == 0:
        return np.zeros(
            (0, descriptor_width * descriptor_width * num_bins), dtype=np.float32
        ), []

    return np.vstack(descriptors), locations


def _sift_features_custom(
    image_bgr: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Full custom SIFT pipeline: Gaussian pyramid, DoG, keypoints, orientation, descriptor.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gaussians = _gaussian_pyramid(gray)
    dogs = _dog_pyramid(gaussians)
    kps = _detect_keypoints(gaussians, dogs)
    oriented_kps = _assign_orientations(gaussians, kps)
    descriptors, locations = _compute_descriptors(gaussians, oriented_kps)
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
    Compute homography H such that dst ~ H * src via DLT.
    src_pts, dst_pts: (N, 2)
    """
    n = src_pts.shape[0]
    A = []
    for i in range(n):
        x, y = src_pts[i, 0], src_pts[i, 1]
        u, v = dst_pts[i, 0], dst_pts[i, 1]
        A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
        A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3, 3)
    return H / (H[2, 2] + 1e-12)


def _ransac_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    num_iterations: int = 1000,
    reproj_thresh: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Manual RANSAC for homography estimation.
    Returns best H and inlier mask.
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
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None:
        return None, None

    # Refine with all inliers
    if np.sum(best_inliers) >= 4:
        H_refined = _compute_homography(src_pts[best_inliers], dst_pts[best_inliers])
    else:
        H_refined = best_H

    return H_refined, best_inliers


def _pairwise_homography_custom(
    img1: np.ndarray, img2: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute homography from img2 to img1 using custom SIFT + RANSAC.
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
    Compute homography from img2 to img1 using OpenCV SIFT + findHomography(RANSAC).
    """
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
# Common stitching function
# --------------------------


def _warp_and_blend(
    images: List[np.ndarray], homographies: List[np.ndarray], prefix: str
) -> str:
    """
    Warp all images into a common canvas using given homographies to image 0,
    and blend by simple averaging.
    """
    _ensure_results_dir()

    h0, w0 = images[0].shape[:2]
    corners = []

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        pts_h = np.hstack([pts, np.ones((4, 1), dtype=np.float32)])
        warped = (homographies[i] @ pts_h.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        corners.append(warped)

    corners = np.vstack(corners)
    x_min = int(np.floor(corners[:, 0].min()))
    x_max = int(np.ceil(corners[:, 0].max()))
    y_min = int(np.floor(corners[:, 1].min()))
    y_max = int(np.ceil(corners[:, 1].max()))

    offset_x = -x_min if x_min < 0 else 0
    offset_y = -y_min if y_min < 0 else 0

    width = x_max - x_min
    height = y_max - y_min

    T = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    acc = np.zeros((height, width, 3), dtype=np.float32)
    weight = np.zeros((height, width, 1), dtype=np.float32)

    for i, img in enumerate(images):
        H = T @ homographies[i]
        warped = cv2.warpPerspective(img.astype(np.float32), H, (width, height))
        mask = (warped.sum(axis=2, keepdims=True) > 0).astype(np.float32)
        acc += warped * mask
        weight += mask

    weight[weight == 0] = 1.0
    pano = (acc / weight).astype(np.uint8)

    filename = _unique_filename(prefix)
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, pano)
    return filename


# --------------------------
# Public API functions
# --------------------------


def stitch_images_custom(images: List[np.ndarray]) -> Dict[str, Any]:
    """
    Custom SIFT + custom RANSAC stitching for a list of images.
    images: list of BGR np.ndarray.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    homographies = [np.eye(3, dtype=np.float32)]
    pair_stats = []

    for i in range(1, len(images)):
        H, stats = _pairwise_homography_custom(images[i - 1], images[i])
        stats["pair"] = f"{i - 1}-{i}"
        pair_stats.append(stats)

        if H is None:
            raise ValueError(
                f"Custom SIFT failed to compute homography for pair {i - 1}-{i}."
            )

        H_to_base = homographies[i - 1] @ H
        homographies.append(H_to_base.astype(np.float32))

    filename = _warp_and_blend(images, homographies, prefix="hw4_custom")

    return {
        "filename": filename,
        "stats": {
            "num_images": len(images),
            "pair_stats": pair_stats,
        },
    }


def stitch_images_opencv(images: List[np.ndarray]) -> Dict[str, Any]:
    """
    OpenCV SIFT + OpenCV findHomography(RANSAC) stitching for a list of images.
    images: list of BGR np.ndarray.
    """
    if len(images) < 2:
        raise ValueError("Need at least two images for stitching.")

    homographies = [np.eye(3, dtype=np.float32)]
    pair_stats = []

    for i in range(1, len(images)):
        H, stats = _pairwise_homography_opencv(images[i - 1], images[i])
        stats["pair"] = f"{i - 1}-{i}"
        pair_stats.append(stats)

        if H is None:
            raise ValueError(
                f"OpenCV SIFT failed to compute homography for pair {i - 1}-{i}."
            )

        H_to_base = homographies[i - 1] @ H
        homographies.append(H_to_base.astype(np.float32))

    filename = _warp_and_blend(images, homographies, prefix="hw4_opencv")

    return {
        "filename": filename,
        "stats": {
            "num_images": len(images),
            "pair_stats": pair_stats,
        },
    }


def store_mobile_panorama(image: np.ndarray) -> Dict[str, Any]:
    """
    Save a single mobile panorama image to static/results/ and return filename.
    image: BGR np.ndarray.
    """
    _ensure_results_dir()
    filename = _unique_filename("hw4_mobile")
    out_path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(out_path, image)
    return {"filename": filename}
