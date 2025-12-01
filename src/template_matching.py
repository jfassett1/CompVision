import os
import cv2
import numpy as np
from datetime import datetime


def load_templates(template_dir):
    """Loads all template images from a directory."""
    templates = []
    for fname in sorted(os.listdir(template_dir)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(template_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                templates.append((fname, img))
    return templates


def _match_corr(scene_gray, templ):
    """Return best template match using correlation coefficient."""
    result = cv2.matchTemplate(scene_gray, templ, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return float(max_val), max_loc


def run_template_matching(
    scene_bgr, templates, threshold=0.8, max_matches=5, result_dir="static/results"
):
    """
    Runs template matching on a scene.
    Returns: (matches_list, output_filename)
    """

    if scene_bgr is None:
        raise ValueError("Scene image cannot be None.")

    scene_gray = cv2.cvtColor(scene_bgr, cv2.COLOR_BGR2GRAY)

    matches = []

    # Match each template
    for name, templ in templates:
        score, top_left = _match_corr(scene_gray, templ)
        if score >= threshold:
            h, w = templ.shape[:2]
            x1, y1 = top_left
            x2, y2 = x1 + w, y1 + h

            matches.append(
                {
                    "template_name": name,
                    "score": score,
                    "top_left": [int(x1), int(y1)],
                    "bottom_right": [int(x2), int(y2)],
                }
            )

    # Sort and trim
    matches = sorted(matches, key=lambda m: m["score"], reverse=True)
    matches = matches[:max_matches]

    # Blur matched regions
    result_img = scene_bgr.copy()
    for m in matches:
        x1, y1 = m["top_left"]
        x2, y2 = m["bottom_right"]

        region = result_img[y1:y2, x1:x2]
        if region.size > 0:
            blurred = cv2.GaussianBlur(region, (25, 25), 0)
            result_img[y1:y2, x1:x2] = blurred

    # Save output
    os.makedirs(result_dir, exist_ok=True)
    out_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    cv2.imwrite(os.path.join(result_dir, out_name), result_img)

    return matches, out_name
