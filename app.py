from src.utils import get_dist
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

from src.template_matching import load_templates, run_template_matching

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("page.html")


TEMPLATE_DIR = "templates_db"
TEMPLATES = load_templates(TEMPLATE_DIR)


@app.route("/2")
def hw2_page():
    return render_template("hw2.html")


@app.route("/api/match_templates", methods=["POST"])
def match_templates():
    if "scene" not in request.files:
        return jsonify({"success": False, "message": "Missing scene image."}), 400

    scene_file = request.files["scene"]
    threshold = float(request.form.get("threshold", 0.8))
    max_matches = int(request.form.get("max_matches", 5))

    # Convert file → cv2 image
    np_arr = np.frombuffer(scene_file.read(), np.uint8)
    scene = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if scene is None:
        return jsonify({"success": False, "message": "Invalid image."}), 400

    matches, filename = run_template_matching(
        scene,
        TEMPLATES,
        threshold=threshold,
        max_matches=max_matches,
        result_dir="static/results",
    )

    return jsonify(
        {
            "success": True,
            "processed_image_url": url_for("static", filename=f"results/{filename}"),
            "matches": matches,
            "message": "Done",
        }
    )


def encode_img(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")


@app.route("/3")
def hw3():
    return render_template("hw3.html")


@app.route("/hw3/process", methods=["POST"])
def hw3_process():
    file = request.files["image"]
    img = Image.open(file.stream).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Gradient magnitude and angle
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    mag_img = cv2.convertScaleAbs(mag)
    ang_img = cv2.applyColorMap(cv2.convertScaleAbs(ang * 255 / 360), cv2.COLORMAP_HSV)

    # Laplacian of Gaussian
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    log = cv2.Laplacian(blur, cv2.CV_64F)
    log_img = cv2.convertScaleAbs(log)

    return jsonify(
        {
            "grad_mag": encode_img(mag_img),
            "grad_ang": encode_img(ang_img),
            "log": encode_img(log_img),
        }
    )


@app.route("/pixel", methods=["POST"])
def pixel_data():
    data = request.get_json()
    x1 = data.get("x1")
    y1 = data.get("y1")
    x2 = data.get("x2")
    y2 = data.get("y2")
    color = data.get("color")  # optional
    # print(f"Received pixel: ({x}, {y}) color={color}")
    pt1 = np.array([x1, y1])
    pt2 = np.array([x2, y2])

    dist = get_dist(pt1, pt2)
    # You can process or log this on the backend

    return jsonify(
        {"status": "success", "message": f"Distance between points: {dist:.2f} in"}
    )


from src.hw4_stitching import (
    stitch_images_custom,
    stitch_images_opencv,
    store_mobile_panorama,
)


@app.route("/4")
def hw4_page():
    return render_template("hw4.html")


@app.route("/api/hw4_custom_stitch", methods=["POST"])
def hw4_custom_stitch():
    files = request.files.getlist("images")
    if not files or len(files) < 2:
        return jsonify(
            {"success": False, "error": "Upload at least 2 images for stitching."}
        )

    images = []
    for f in files:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return jsonify({"success": False, "error": "Could not decode enough images."})

    try:
        result = stitch_images_custom(images)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    image_url = url_for("static", filename=f"results/{result['filename']}")
    return jsonify(
        {
            "success": True,
            "image_url": image_url,
            "stats": result.get("stats", {}),
        }
    )


@app.route("/api/hw4_opencv_stitch", methods=["POST"])
def hw4_opencv_stitch():
    files = request.files.getlist("images")
    if not files or len(files) < 2:
        return jsonify(
            {"success": False, "error": "Upload at least 2 images for stitching."}
        )

    images = []
    for f in files:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if len(images) < 2:
        return jsonify({"success": False, "error": "Could not decode enough images."})

    try:
        result = stitch_images_opencv(images)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

    image_url = url_for("static", filename=f"results/{result['filename']}")
    return jsonify(
        {
            "success": True,
            "image_url": image_url,
            "stats": result.get("stats", {}),
        }
    )


@app.route("/api/hw4_mobile_panorama", methods=["POST"])
def hw4_mobile_panorama():
    f = request.files.get("image", None)
    if f is None:
        return jsonify({"success": False, "error": "No image file provided."})

    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Could not decode image."})

    result = store_mobile_panorama(img)
    image_url = url_for("static", filename=f"results/{result['filename']}")
    return jsonify(
        {
            "success": True,
            "image_url": image_url,
        }
    )


from src.hw5_tracking import (
    decode_video_file,
    track_marker,
    track_markerless,
    track_sam2_from_npz,
    save_frame_sequence,
    summarize_tracking,
)


@app.route("/5")
def hw5_page():
    return render_template("hw5.html")


@app.route("/api/hw5_marker", methods=["POST"])
def hw5_marker():
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"success": False, "error": "No video file uploaded."}), 400

    try:
        frames = decode_video_file(video_file)
        results, filenames = track_marker(frames, marker_type="aruco")
        stats = summarize_tracking(results)
        frame_urls = [url_for("static", filename=fname) for fname in filenames]

        return jsonify(
            {
                "success": True,
                "frames": frame_urls,
                "stats": stats,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hw5_markerless", methods=["POST"])
def hw5_markerless():
    video_file = request.files.get("video")
    if not video_file:
        return jsonify({"success": False, "error": "No video file uploaded."}), 400

    try:
        frames = decode_video_file(video_file)
        # No explicit init_bbox from UI; using automatic initialization inside tracker.
        results, filenames = track_markerless(frames)
        stats = summarize_tracking(results)
        frame_urls = [url_for("static", filename=fname) for fname in filenames]

        return jsonify(
            {
                "success": True,
                "frames": frame_urls,
                "stats": stats,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/hw5_sam2", methods=["POST"])
def hw5_sam2():
    video_file = request.files.get("video")
    npz_file = request.files.get("npz")

    if not video_file:
        return jsonify({"success": False, "error": "No video file uploaded."}), 400
    if not npz_file:
        return jsonify({"success": False, "error": "No NPZ file uploaded."}), 400

    try:
        frames = decode_video_file(video_file)
        # track_sam2_from_npz can consume a FileStorage-like object directly
        results, filenames = track_sam2_from_npz(frames, npz_file)
        stats = summarize_tracking(results)
        frame_urls = [url_for("static", filename=fname) for fname in filenames]

        return jsonify(
            {
                "success": True,
                "frames": frame_urls,
                "stats": stats,
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


from src.hw7_pose import (
    decode_video_file,
    process_pose_video,
    save_pose_csv,
    save_processed_frames,
)


@app.route("/7")
def hw7_page():
    return render_template("hw7.html")


@app.route("/api/hw7_pose", methods=["POST"])
def hw7_pose():
    video_file = request.files.get("video")
    if video_file is None:
        return jsonify(success=False, error="No video uploaded."), 400

    try:
        frames = decode_video_file(video_file)
        if not frames:
            return jsonify(success=False, error="Could not decode frames."), 400

        processed_frames, keypoints = process_pose_video(frames)

        preview_file = save_processed_frames(processed_frames)
        csv_file = save_pose_csv(keypoints)

        # Stats
        num_frames = len(frames)
        num_detect = sum(
            1
            for f in keypoints
            if f["pose"] or f["hands"]["left"] or f["hands"]["right"]
        )

        pose_joints = set()
        hand_joints = set()
        for f in keypoints:
            pose_joints.update(f["pose"].keys())
            hand_joints.update(f["hands"]["left"].keys())
            hand_joints.update(f["hands"]["right"].keys())

        stats = {
            "num_frames": num_frames,
            "num_detected_frames": num_detect,
            "num_pose_joints": len(pose_joints),
            "num_hand_joints": len(hand_joints),
        }

        return jsonify(
            success=True,
            preview_image_url=url_for("static", filename="results/" + preview_file),
            csv_url=url_for("static", filename="results/" + csv_file),
            stats=stats,
        )

    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
