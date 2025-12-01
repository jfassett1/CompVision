import cv2
import numpy as np
from pathlib import Path

# Configuration
input_dir = (
    Path(__file__).parent.parent / "data" / "dataset2"
)  # folder containing your dataset images
output_dir = (
    Path(__file__).parent.parent / "data" / "dataset2_output"
)  # folder containing your dataset images
output_dir.mkdir(exist_ok=True)


# -----------------------------
# 1. Gradient Computation
# -----------------------------
def compute_gradients(img_gray):
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    mag_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ang_vis = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_vis, ang_vis, magnitude, gx, gy


# -----------------------------
# 2. Laplacian of Gaussian
# -----------------------------
def compute_log(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    log_vis = cv2.convertScaleAbs(log)
    return log_vis


# -----------------------------
# 3. Edge Keypoints
# -----------------------------
def detect_edge_keypoints(magnitude, grad_thresh=80):
    mag_norm = (magnitude / magnitude.max()) * 255
    edges = (mag_norm > grad_thresh).astype(np.uint8) * 255
    return edges


# -----------------------------
# 4. Corner Keypoints
# -----------------------------
def detect_corner_keypoints(img_gray, window_size=3, k=0.04, thresh=0.01):
    gray = np.float32(img_gray)
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    Ixx = cv2.GaussianBlur(Ix * Ix, (window_size, window_size), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (window_size, window_size), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (window_size, window_size), 0)
    det = Ixx * Iyy - Ixy**2
    trace = Ixx + Iyy
    R = det - k * (trace**2)
    R_norm = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    corners = np.zeros_like(R_norm, dtype=np.uint8)
    corners[R_norm > thresh] = 255
    return corners


# -----------------------------
# 5. Main Processing Loop
# -----------------------------
for img_path in input_dir.glob("*.*"):
    print(f"Processing {img_path.name}...")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Skipping {img_path.name} (invalid image).")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Gradient magnitude and angle
    mag_vis, ang_vis, magnitude, gx, gy = compute_gradients(gray)
    cv2.imwrite(str(output_dir / f"{img_path.stem}_grad_mag.jpg"), mag_vis)
    cv2.imwrite(str(output_dir / f"{img_path.stem}_grad_ang.jpg"), ang_vis)

    # Step 2: Laplacian of Gaussian
    log_vis = compute_log(gray)
    cv2.imwrite(str(output_dir / f"{img_path.stem}_log.jpg"), log_vis)

    # Step 3: Edge keypoints (based on gradient threshold)
    edge_points = detect_edge_keypoints(magnitude)
    edges_vis = img.copy()
    edges_vis[edge_points > 0] = [0, 0, 255]  # red overlay
    cv2.imwrite(str(output_dir / f"{img_path.stem}_edges.jpg"), edges_vis)

    # Step 4: Corner keypoints (Harris-based)
    corner_points = detect_corner_keypoints(gray)
    corners_vis = img.copy()
    corners_vis[corner_points > 0] = [0, 255, 0]  # green overlay
    cv2.imwrite(str(output_dir / f"{img_path.stem}_corners.jpg"), corners_vis)

    # Step 5: Combined visualization (original | gradient mag | LoG)
    comparison = np.hstack(
        [
            cv2.resize(gray, (256, 256)),
            cv2.resize(mag_vis, (256, 256)),
            cv2.resize(log_vis, (256, 256)),
        ]
    )
    cv2.imwrite(str(output_dir / f"{img_path.stem}_comparison.jpg"), comparison)

print("\nAll outputs saved in:", output_dir)
