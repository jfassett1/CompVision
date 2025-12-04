import os
import time
import math

import cv2
import numpy as np

# --------------------------------------------------------------------
# Camera intrinsics and calibration resolution
# --------------------------------------------------------------------

fx, fy = 1872.9844024046715, 958.1184951213671
cx, cy = 1869.1125345607425, 543.8823586167894

K = np.array(
    [
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

TARGET_WIDTH = 1920
TARGET_HEIGHT = 1280


def resize_to_calibrated_resolution(image_bgr):
    """
    Resize input image to 1920x1280 (width x height).
    Returns (resized_image, scale_x, scale_y).

    scale_x = TARGET_WIDTH / original_width
    scale_y = TARGET_HEIGHT / original_height
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to resize_to_calibrated_resolution.")

    h, w = image_bgr.shape[:2]
    resized = cv2.resize(
        image_bgr,
        (TARGET_WIDTH, TARGET_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    scale_x = TARGET_WIDTH / float(w)
    scale_y = TARGET_HEIGHT / float(h)
    return resized, scale_x, scale_y


# --------------------------------------------------------------------
# Real-world length from known distance (Problem 1 core)
# --------------------------------------------------------------------


def compute_real_world_length_from_distance(
    u1,
    v1,
    u2,
    v2,
    Z,
    axis="auto",
):
    """
    Given two image points (u1,v1), (u2,v2) in pixels on the resized image,
    and known distance Z (in same units as desired real-world length, e.g. meters),
    compute the real-world length of the segment between them using the
    perspective projection model and the fixed intrinsics.

    Pinhole model:
        u = fx * X / Z + cx
        v = fy * Y / Z + cy

    For a segment:

        Horizontal:
            Δu = |u2 - u1|
            X_length ≈ (Z / fx) * Δu

        Vertical:
            Δv = |v2 - v1|
            Y_length ≈ (Z / fy) * Δv

        General orientation:
            d_pix = sqrt((Δu)^2 + (Δv)^2)
            f_eff = (fx + fy) / 2
            L_real ≈ (Z / f_eff) * d_pix
    """
    u1 = float(u1)
    v1 = float(v1)
    u2 = float(u2)
    v2 = float(v2)
    Z = float(Z)

    if Z <= 0:
        raise ValueError("Distance Z must be positive.")

    du = u2 - u1
    dv = v2 - v1
    abs_du = abs(du)
    abs_dv = abs(dv)
    d_pix = math.hypot(du, dv)

    if d_pix <= 0:
        raise ValueError("Degenerate segment: the two points are identical.")

    axis_used = axis
    if axis == "auto":
        if abs_du >= 2.0 * abs_dv:
            axis_used = "horizontal"
        elif abs_dv >= 2.0 * abs_du:
            axis_used = "vertical"
        else:
            axis_used = "general"

    Lx = (Z / fx) * abs_du
    Ly = (Z / fy) * abs_dv

    if axis_used == "horizontal":
        L_real = Lx
    elif axis_used == "vertical":
        L_real = Ly
    else:
        f_eff = 0.5 * (fx + fy)
        L_real = (Z / f_eff) * d_pix

    return {
        "L_real": float(L_real),
        "Lx": float(Lx),
        "Ly": float(Ly),
        "axis_used": axis_used,
        "d_pix": float(d_pix),
        "du": float(du),
        "dv": float(dv),
        "Z": float(Z),
    }


# --------------------------------------------------------------------
# Distance from known real-world length (kept for offline use if needed)
# --------------------------------------------------------------------


def compute_distance_from_real_world_length(
    u1,
    v1,
    u2,
    v2,
    L_real,
    axis="auto",
):
    """
    Inverse of compute_real_world_length_from_distance.
    Not used by the web app, but available for offline validation.
    """
    u1 = float(u1)
    v1 = float(v1)
    u2 = float(u2)
    v2 = float(v2)
    L_real = float(L_real)

    if L_real <= 0:
        raise ValueError("Real-world length L_real must be positive.")

    du = u2 - u1
    dv = v2 - v1
    abs_du = abs(du)
    abs_dv = abs(dv)
    d_pix = math.hypot(du, dv)

    if d_pix <= 0:
        raise ValueError("Degenerate segment: the two points are identical.")

    axis_used = axis
    if axis == "auto":
        if abs_du >= 2.0 * abs_dv:
            axis_used = "horizontal"
        elif abs_dv >= 2.0 * abs_du:
            axis_used = "vertical"
        else:
            axis_used = "general"

    if axis_used == "horizontal":
        if abs_du == 0:
            raise ValueError("Horizontal axis chosen but Δu is zero.")
        Z = fx * L_real / abs_du
    elif axis_used == "vertical":
        if abs_dv == 0:
            raise ValueError("Vertical axis chosen but Δv is zero.")
        Z = fy * L_real / abs_dv
    else:
        f_eff = 0.5 * (fx + fy)
        Z = f_eff * L_real / d_pix

    return {
        "Z": float(Z),
        "axis_used": axis_used,
        "d_pix": float(d_pix),
        "du": float(du),
        "dv": float(dv),
        "L_real": float(L_real),
    }


# --------------------------------------------------------------------
# Image decoding and overlay utilities
# --------------------------------------------------------------------


def decode_image_file(file_storage):
    """
    Flask FileStorage -> BGR NumPy array, resized to 1920x1280.
    Returns (image_resized, scale_x, scale_y).

    scale_x, scale_y are the factors from original image to resized:
        u_resized = u_original * scale_x
        v_resized = v_original * scale_y
    """
    if file_storage is None:
        raise ValueError("No image file provided.")

    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Invalid image file.")

    return resize_to_calibrated_resolution(image)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def draw_segment_and_save(
    image_bgr,
    u1,
    v1,
    u2,
    v2,
    result_dir="static/results",
    prefix="hw1_segment",
):
    """
    Draw the user-selected segment on the resized image, save it, and return filename.

    image_bgr is assumed to already be resized to 1920x1280.
    (u1, v1), (u2, v2) are in the same coordinate system (pixels).
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to draw_segment_and_save.")

    _ensure_dir(result_dir)

    img_draw = image_bgr.copy()

    p1 = (int(round(u1)), int(round(v1)))
    p2 = (int(round(u2)), int(round(v2)))

    cv2.line(img_draw, p1, p2, (0, 0, 255), thickness=2)
    cv2.circle(img_draw, p1, 5, (0, 255, 0), thickness=-1)
    cv2.circle(img_draw, p2, 5, (0, 255, 0), thickness=-1)

    timestamp = int(time.time() * 1000)
    filename = f"{prefix}_{timestamp}.png"
    out_path = os.path.join(result_dir, filename)

    cv2.imwrite(out_path, img_draw)

    return filename
