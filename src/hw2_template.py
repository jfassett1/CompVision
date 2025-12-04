import os
import time
import uuid

import cv2
import numpy as np


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


# =========================
# 1. TEMPLATE MATCHING API
# =========================


def load_template_db(template_dir=None):
    """
    AUTO-GENERATES 10 SIMPLE SYNTHETIC TEMPLATES.
    Saves them for visualization and returns them as grayscale arrays.
    """
    out_dir = "static/results/hw2_generated_templates"
    os.makedirs(out_dir, exist_ok=True)

    templates = []
    size = 80  # template size

    def save_and_append(name, img):
        fname = f"{name}.png"
        cv2.imwrite(os.path.join(out_dir, fname), img)
        templates.append((name, img))

    # Template 1: Filled square
    img = np.zeros((size, size), np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), 255, -1)
    save_and_append("square", img)

    # Template 2: Hollow square
    img = np.zeros((size, size), np.uint8)
    cv2.rectangle(img, (10, 10), (70, 70), 255, 3)
    save_and_append("square_hollow", img)

    # Template 3: Filled circle
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (40, 40), 25, 255, -1)
    save_and_append("circle", img)

    # Template 4: Hollow circle
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, (40, 40), 25, 255, 3)
    save_and_append("circle_hollow", img)

    # Template 5: Horizontal stripes
    img = np.zeros((size, size), np.uint8)
    img[::10, :] = 255
    save_and_append("stripes_h", img)

    # Template 6: Vertical stripes
    img = np.zeros((size, size), np.uint8)
    img[:, ::10] = 255
    save_and_append("stripes_v", img)

    # Template 7: Diagonal slash
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (0, 0), (79, 79), 255, 5)
    save_and_append("diag_fwd", img)

    # Template 8: Diagonal backslash
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (79, 0), (0, 79), 255, 5)
    save_and_append("diag_bwd", img)

    # Template 9: Cross (X)
    img = np.zeros((size, size), np.uint8)
    cv2.line(img, (0, 0), (79, 79), 255, 3)
    cv2.line(img, (79, 0), (0, 79), 255, 3)
    save_and_append("cross_x", img)

    # Template 10: Gradient
    img = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    save_and_append("gradient", img)

    return templates


def match_templates_in_scene(scene_bgr, templates, threshold=0.8, max_matches=10):
    """
    Perform correlation-based template matching for each template against the scene.

    Returns a list of match dicts:
    [
      {
        "template_name": ...,
        "score": ...,
        "top_left": [x1, y1],
        "bottom_right": [x2, y2]
      },
      ...
    ]
    """
    if scene_bgr is None or len(templates) == 0:
        return []

    if len(scene_bgr.shape) == 3:
        scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)
    else:
        scene_gray = scene_bgr.copy()

    h_scene, w_scene = scene_gray.shape[:2]
    matches = []

    for template_name, template_gray in templates:
        if template_gray is None:
            continue
        h_temp, w_temp = template_gray.shape[:2]

        if h_temp > h_scene or w_temp > w_scene:
            # Template cannot be matched if larger than scene
            continue

        res = cv2.matchTemplate(scene_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        score = float(max_val)
        if score >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + w_temp, top_left[1] + h_temp)
            matches.append(
                {
                    "template_name": template_name,
                    "score": score,
                    "top_left": [int(top_left[0]), int(top_left[1])],
                    "bottom_right": [int(bottom_right[0]), int(bottom_right[1])],
                }
            )

    # Sort matches by score (descending) and keep top max_matches
    matches.sort(key=lambda m: m["score"], reverse=True)
    if max_matches is not None and max_matches > 0:
        matches = matches[:max_matches]

    return matches


def blur_matched_regions(
    scene_bgr, matches, result_dir="static/results", prefix="hw2_match"
):
    """
    Blur matched regions in the scene and save the resulting image.

    Returns the output filename (basename only).
    """
    if scene_bgr is None:
        return None

    _ensure_dir(result_dir)

    output = scene_bgr.copy()
    h_img, w_img = output.shape[:2]

    for match in matches:
        tl = match.get("top_left", [0, 0])
        br = match.get("bottom_right", [0, 0])
        x1, y1 = int(tl[0]), int(tl[1])
        x2, y2 = int(br[0]), int(br[1])

        # Clamp coordinates
        x1 = max(0, min(w_img - 1, x1))
        x2 = max(0, min(w_img, x2))
        y1 = max(0, min(h_img - 1, y1))
        y2 = max(0, min(h_img, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        ksize = (31, 31)
        roi = output[y1:y2, x1:x2]
        blurred_roi = cv2.GaussianBlur(roi, ksize, sigmaX=0, sigmaY=0)
        output[y1:y2, x1:x2] = blurred_roi

        # Draw a visible bounding box to show the blurred region
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),  # bright red
            3,  # line thickness
        )
        roi = output[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Apply strong blur to the region

    unique_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    filename = f"{prefix}_{unique_tag}.png"
    out_path = os.path.join(result_dir, filename)
    cv2.imwrite(out_path, output)

    return filename


def run_template_matching_pipeline(
    scene_bgr,
    templates,
    threshold=0.8,
    max_matches=10,
    result_dir="static/results",
):
    """
    Convenience pipeline: run matching and blur matched regions.

    Returns (matches, output_filename).
    """
    matches = match_templates_in_scene(
        scene_bgr,
        templates,
        threshold=threshold,
        max_matches=max_matches,
    )
    output_filename = blur_matched_regions(
        scene_bgr,
        matches,
        result_dir=result_dir,
        prefix="hw2_match",
    )
    return matches, output_filename


# ==========================================
# 2. GAUSSIAN BLUR + FOURIER DEBLURRING API
# ==========================================


def apply_gaussian_blur(
    image_bgr, ksize=(15, 15), sigma=2.0, result_dir="static/results"
):
    """
    Apply Gaussian blur to an image.

    Operates in grayscale for L and L_b.

    Returns:
        blurred_gray (np.ndarray),
        original_filename (basename),
        blurred_filename (basename)
    """
    _ensure_dir(result_dir)

    if len(image_bgr.shape) == 3:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_bgr.copy()

    if sigma <= 0:
        sigma = 1.0

    blurred = cv2.GaussianBlur(gray, ksize, sigmaX=sigma, sigmaY=sigma)

    unique_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    original_filename = f"hw2_fourier_original_{unique_tag}.png"
    blurred_filename = f"hw2_fourier_blurred_{unique_tag}.png"

    cv2.imwrite(os.path.join(result_dir, original_filename), gray)
    cv2.imwrite(os.path.join(result_dir, blurred_filename), blurred)

    return blurred, original_filename, blurred_filename


def fourier_deblur(blurred_gray, sigma, eps=1e-3, result_dir="static/results"):
    """
    Approximate deblurring in the Fourier domain given a Gaussian blur model.

    Returns:
        recovered_filename (basename).
    """
    _ensure_dir(result_dir)

    if blurred_gray is None:
        return None

    if sigma <= 0:
        sigma = 1.0

    # Convert to float32 for FFT
    img = blurred_gray.astype(np.float32) / 255.0
    h, w = img.shape[:2]

    # Create Gaussian PSF in spatial domain
    k = int(6 * sigma + 1)
    if k % 2 == 0:
        k += 1
    g1d = cv2.getGaussianKernel(k, sigma)
    psf = g1d @ g1d.T  # 2D Gaussian

    # Embed PSF into same size as image (top-left), then center it via roll
    psf_full = np.zeros_like(img)
    kh, kw = psf.shape
    psf_full[:kh, :kw] = psf
    psf_full = np.roll(psf_full, -kh // 2, axis=0)
    psf_full = np.roll(psf_full, -kw // 2, axis=1)

    # Compute FFTs
    B = np.fft.fft2(img)
    H = np.fft.fft2(psf_full)

    # Avoid division by zero
    H_abs = np.abs(H)
    H_safe = H.copy()
    H_safe[H_abs < eps] = eps

    # Simple deconvolution
    L_hat = B / H_safe
    l_rec = np.fft.ifft2(L_hat)
    l_rec = np.real(l_rec)

    # Normalize / clip
    l_rec = np.clip(l_rec, 0.0, 1.0)
    l_rec_uint8 = (l_rec * 255.0).astype(np.uint8)

    unique_tag = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    recovered_filename = f"hw2_fourier_recovered_{unique_tag}.png"
    cv2.imwrite(os.path.join(result_dir, recovered_filename), l_rec_uint8)

    return recovered_filename


def run_fourier_deblur_pipeline(
    image_bgr,
    ksize=None,
    sigma=2.0,
    result_dir="static/results",
):
    """
    Combined pipeline:
      1) Convert L to grayscale and blur to get L_b.
      2) Fourier deblur L_b to approximate L.

    Returns:
        (original_filename, blurred_filename, recovered_filename)
    """
    if ksize is None:
        # Derive reasonable kernel size from sigma
        k = int(6 * sigma + 1)
        if k % 2 == 0:
            k += 1
        ksize = (k, k)

    blurred_gray, original_fname, blurred_fname = apply_gaussian_blur(
        image_bgr, ksize=ksize, sigma=sigma, result_dir=result_dir
    )

    recovered_fname = fourier_deblur(
        blurred_gray,
        sigma=sigma,
        eps=1e-3,
        result_dir=result_dir,
    )

    return original_fname, blurred_fname, recovered_fname
