import os
import csv
import uuid
import tempfile
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp


RESULTS_DIR = os.path.join("static", "results")


def _ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def decode_video_file(file_storage):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_path = tmp.name
        file_storage.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    os.remove(temp_path)
    return frames


def _get_timestamp_prefix(prefix="hw7_pose"):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:8]
    return f"{prefix}_{now}_{rand}"


# ---------------------------------------------------------
# MEDIAPIPE-ONLY POSE + HAND TRACKING
# ---------------------------------------------------------
def process_pose_video(frames):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    processed = []
    keypoint_sequence = []

    pose = mp_pose.Pose(
        static_image_mode=False, model_complexity=1, enable_segmentation=False
    )
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        for idx, frame in enumerate(frames):
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pose_result = pose.process(rgb)
            hands_result = hands.process(rgb)

            out = frame.copy()

            frame_data = {
                "frame_index": idx,
                "pose": {},
                "hands": {"left": {}, "right": {}},
            }

            # --------------------------
            # Pose landmarks
            # --------------------------
            if pose_result.pose_landmarks:
                mp_drawing.draw_landmarks(
                    out,
                    pose_result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style(),
                )

                for lm_enum in mp_pose.PoseLandmark:
                    lm = pose_result.pose_landmarks.landmark[lm_enum]
                    name = lm_enum.name.lower()
                    frame_data["pose"][name] = [
                        float(lm.x * w),
                        float(lm.y * h),
                        float(lm.z),
                        float(lm.visibility),
                    ]

            # --------------------------
            # Hand landmarks
            # --------------------------
            if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
                for hls, handedness in zip(
                    hands_result.multi_hand_landmarks, hands_result.multi_handedness
                ):
                    label = handedness.classification[0].label.lower()

                    mp_drawing.draw_landmarks(out, hls, mp_hands.HAND_CONNECTIONS)

                    for lm_enum in mp_hands.HandLandmark:
                        lm = hls.landmark[lm_enum]
                        name = lm_enum.name.lower()
                        frame_data["hands"][label][name] = [
                            float(lm.x * w),
                            float(lm.y * h),
                            float(lm.z),
                            1.0,
                        ]

            processed.append(out)
            keypoint_sequence.append(frame_data)

    finally:
        pose.close()
        hands.close()

    return processed, keypoint_sequence


# ---------------------------------------------------------
# SAVING OUTPUT
# ---------------------------------------------------------
def save_processed_frames(frames, prefix="hw7_pose"):
    _ensure_results_dir()
    filename = f"{_get_timestamp_prefix(prefix)}_preview.png"
    path = os.path.join(RESULTS_DIR, filename)
    cv2.imwrite(path, frames[0])
    return filename


def save_pose_csv(keypoint_sequence, csv_prefix="hw7_pose"):
    _ensure_results_dir()
    filename = f"{_get_timestamp_prefix(csv_prefix)}.csv"
    path = os.path.join(RESULTS_DIR, filename)

    fieldnames = [
        "frame_index",
        "joint_group",
        "joint_side",
        "joint_name",
        "x",
        "y",
        "z",
        "confidence",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for fdata in keypoint_sequence:
            fi = fdata["frame_index"]

            # Pose joints
            for name, vals in fdata["pose"].items():
                x, y, z, c = vals
                writer.writerow(
                    {
                        "frame_index": fi,
                        "joint_group": "pose",
                        "joint_side": "",
                        "joint_name": name,
                        "x": x,
                        "y": y,
                        "z": z,
                        "confidence": c,
                    }
                )

            # Hands
            for side in ["left", "right"]:
                for name, vals in fdata["hands"][side].items():
                    x, y, z, c = vals
                    writer.writerow(
                        {
                            "frame_index": fi,
                            "joint_group": "hand",
                            "joint_side": side,
                            "joint_name": name,
                            "x": x,
                            "y": y,
                            "z": z,
                            "confidence": c,
                        }
                    )

    return filename
