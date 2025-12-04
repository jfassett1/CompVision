import numpy as np
import cv2

fx, fy = 1872.9844024046715, 958.1184951213671
cx, cy = 1869.1125345607425, 543.8823586167894

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# Two pixel points (u, v)
# pt1 = np.array([800, 400], dtype=np.float32)
# pt2 = np.array([850, 420], dtype=np.float32)

# Known depth from camera (in meters)
Z = 1.5  # e.g., 1.5 meters from camera


# Compute (X, Y) for both points
def pixel_to_world(u, v, K, Z):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z])


def get_dist(pt1, pt2):
    P1 = pixel_to_world(*pt1, K, Z)
    P2 = pixel_to_world(*pt2, K, Z)
    return np.linalg.norm(P1 - P2)
