from src.utils import get_dist
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import cv2
import base64
import uuid
import os

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("page.html")


TEMPLATE_DIR = "templates_db"


@app.route("/static/results/hw2_generated_templates/")
def list_hw2_templates():
    path = os.path.join(app.static_folder, "results/hw2_generated_templates")
    files = sorted(os.listdir(path))
    links = [f'<a href="{fname}">{fname}</a>' for fname in files]
    return "<br>".join(links)


# ---- IMPORTS ----
from src.hw2_template import (
    load_template_db,
    run_template_matching_pipeline,
    run_fourier_deblur_pipeline,
)

# Preload template database for HW2

TEMPLATE_DB_HW2 = load_template_db()


# ---- PAGE ROUTE ----
@app.route("/2")
def hw2_page():
    return render_template("hw2.html")


# ---- API ROUTES ----
@app.route("/api/hw2_match_templates", methods=["POST"])
def hw2_match_templates():
    file = request.files.get("scene", None)
    if file is None or file.filename == "":
        return jsonify(success=False, message="No scene image uploaded.")

    # Parse parameters
    threshold_str = request.form.get("threshold", "").strip()
    max_matches_str = request.form.get("max_matches", "").strip()

    try:
        threshold = float(threshold_str) if threshold_str != "" else 0.8
    except ValueError:
        threshold = 0.8

    try:
        max_matches = int(max_matches_str) if max_matches_str != "" else 5
    except ValueError:
        max_matches = 5

    # Decode image to NumPy array (BGR)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    scene_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if scene_bgr is None:
        return jsonify(success=False, message="Failed to decode scene image.")

    if not TEMPLATE_DB_HW2:
        return jsonify(
            success=False, message="Template database for HW2 is empty or not loaded."
        )

    matches, out_fname = run_template_matching_pipeline(
        scene_bgr,
        TEMPLATE_DB_HW2,
        threshold=threshold,
        max_matches=max_matches,
        result_dir=os.path.join(app.static_folder, "results"),
    )

    if out_fname is None:
        return jsonify(
            success=False, message="Failed to generate processed image.", matches=[]
        )

    processed_image_url = url_for("static", filename=f"results/{out_fname}")

    # Ensure matches are JSON-serializable (convert np types to Python types)
    safe_matches = []
    for m in matches:
        safe_matches.append(
            {
                "template_name": str(m.get("template_name", "")),
                "score": float(m.get("score", 0.0)),
                "top_left": [
                    int(m.get("top_left", [0, 0])[0]),
                    int(m.get("top_left", [0, 0])[1]),
                ],
                "bottom_right": [
                    int(m.get("bottom_right", [0, 0])[0]),
                    int(m.get("bottom_right", [0, 0])[1]),
                ],
            }
        )

    msg = f"Template matching completed with {len(safe_matches)} match(es) above threshold {threshold:.2f}."
    return jsonify(
        success=True,
        processed_image_url=processed_image_url,
        matches=safe_matches,
        message=msg,
    )


@app.route("/api/hw2_fourier_deblur", methods=["POST"])
def hw2_fourier_deblur():
    file = request.files.get("image", None)
    if file is None or file.filename == "":
        return jsonify(success=False, message="No image uploaded for Fourier deblur.")

    sigma_str = request.form.get("sigma", "").strip()
    try:
        sigma = float(sigma_str) if sigma_str != "" else 2.0
    except ValueError:
        sigma = 2.0

    if sigma <= 0:
        sigma = 1.0

    # Decode image to NumPy array (BGR)
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify(success=False, message="Failed to decode input image.")

    orig_fname, blurred_fname, recovered_fname = run_fourier_deblur_pipeline(
        image_bgr,
        ksize=None,
        sigma=sigma,
        result_dir=os.path.join(app.static_folder, "results"),
    )

    if orig_fname is None or blurred_fname is None or recovered_fname is None:
        return jsonify(success=False, message="Fourier blur/deblur pipeline failed.")

    original_url = url_for("static", filename=f"results/{orig_fname}")
    blurred_url = url_for("static", filename=f"results/{blurred_fname}")
    recovered_url = url_for("static", filename=f"results/{recovered_fname}")

    msg = f"Gaussian blur (sigma={sigma:.2f}) and Fourier deblur completed."
    return jsonify(
        success=True,
        original_url=original_url,
        blurred_url=blurred_url,
        recovered_url=recovered_url,
        message=msg,
    )


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


# Imports
from src.hw3 import (
    compute_gradients_and_log,
    run_edge_corner_detection,
    find_object_boundary,
    segment_object_with_aruco,
)

# Routes


@app.route("/3")
def hw3_page():
    return render_template("hw3.html")


@app.route("/api/hw3_grad_log", methods=["POST"])
def hw3_grad_log():
    files = request.files.getlist("images")
    images = []
    for f in files:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        return jsonify({"error": "No valid images uploaded."}), 400

    results = compute_gradients_and_log(images)

    # Convert internal paths (relative to static/) to URLs
    out = []
    for item in results:
        out.append(
            {
                "grad_mag_url": url_for("static", filename=item["grad_mag_path"]),
                "grad_angle_url": url_for("static", filename=item["grad_angle_path"]),
                "log_url": url_for("static", filename=item["log_path"]),
                "mean_grad_mag": item.get("mean_grad_mag", 0.0),
                "index": item.get("index", 0),
            }
        )

    return jsonify({"results": out})


@app.route("/api/hw3_edge_corner", methods=["POST"])
def hw3_edge_corner():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded."}), 400

    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image."}), 400

    result = run_edge_corner_detection(img)

    return jsonify(
        {
            "edge_overlay_url": url_for("static", filename=result["edge_overlay_path"]),
            "corner_overlay_url": url_for(
                "static", filename=result["corner_overlay_path"]
            ),
            "edge_count": result["edge_count"],
            "corner_count": result["corner_count"],
        }
    )


@app.route("/api/hw3_boundary", methods=["POST"])
def hw3_boundary():
    f = request.files.get("image")
    if f is None:
        return jsonify({"error": "No image uploaded."}), 400

    file_bytes = np.frombuffer(f.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image."}), 400

    result = find_object_boundary(img)

    response = {
        "boundary_overlay_url": url_for(
            "static", filename=result["boundary_overlay_path"]
        ),
        "area": result.get("area", 0.0),
        "perimeter": result.get("perimeter", 0.0),
        "bbox": result.get("bbox"),
    }

    mask_path = result.get("mask_path")
    if mask_path:
        response["mask_url"] = url_for("static", filename=mask_path)
    else:
        response["mask_url"] = None

    return jsonify(response)


@app.route("/api/hw3_aruco_seg", methods=["POST"])
def hw3_aruco_seg():
    files = request.files.getlist("images")
    images = []
    for f in files:
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    if not images:
        return jsonify({"error": "No valid images uploaded."}), 400

    result = segment_object_with_aruco(images)

    overlay_urls = [
        url_for("static", filename=p) for p in result.get("overlay_paths", [])
    ]

    return jsonify(
        {
            "overlay_urls": overlay_urls,
            "summary": result.get("summary", {}),
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
        {"status": "success", "message": f"Distance between points: {dist:.2f} m"}
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
    summarize_tracking,
)


@app.route("/5")
def hw5_page():
    return render_template("hw5.html")


@app.route("/api/hw5_marker", methods=["POST"])
def hw5_marker():
    video = request.files.get("video")
    if not video:
        return jsonify({"success": False, "error": "No video"}), 400

    frames = decode_video_file(video)
    results, video_rel = track_marker(frames)
    stats = summarize_tracking(results)

    return jsonify(
        {
            "success": True,
            "video": url_for("static", filename=video_rel),
            "stats": stats,
        }
    )


@app.route("/api/hw5_markerless", methods=["POST"])
def hw5_markerless():
    video = request.files.get("video")
    if not video:
        return jsonify({"success": False, "error": "No video"}), 400

    frames = decode_video_file(video)
    results, video_rel = track_markerless(frames)
    stats = summarize_tracking(results)

    return jsonify(
        {
            "success": True,
            "video": url_for("static", filename=video_rel),
            "stats": stats,
        }
    )


@app.route("/api/hw5_sam2", methods=["POST"])
def hw5_sam2():
    video = request.files.get("video")
    npz = request.files.get("npz")

    if not video:
        return jsonify({"success": False, "error": "No video"}), 400
    if not npz:
        return jsonify({"success": False, "error": "No NPZ"}), 400

    frames = decode_video_file(video)
    results, video_rel = track_sam2_from_npz(frames, npz)
    stats = summarize_tracking(results)

    return jsonify(
        {
            "success": True,
            "video": url_for("static", filename=video_rel),
            "stats": stats,
        }
    )


from src.hw7_pose import (
    decode_video_file,
    process_pose_video,
    save_pose_csv,
    save_processed_frames,
    append_pose_csv_stream,
)


@app.route("/7")
def hw7_page():
    return render_template("hw7.html")


@app.route("/api/hw7_pose_frame", methods=["POST"])
def hw7_pose_frame():
    file = request.files.get("frame")
    run_id = request.form.get("run_id")

    if file is None:
        return jsonify(success=False, error="No frame uploaded."), 400
    if not run_id:
        return jsonify(success=False, error="Missing run_id."), 400

    file_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(success=False, error="Failed to decode frame."), 400

    try:
        # Process single frame as a 1-element list
        processed_frames, keypoints = process_pose_video([frame])
        out = processed_frames[0]

        # Save preview image (you can overwrite per run or create new each time)
        img_name = f"hw7_rt_{run_id}_{uuid.uuid4().hex[:6]}.png"
        img_path = os.path.join("static", "results", img_name)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, out)

        # Append to CSV for this run_id
        csv_filename = append_pose_csv_stream(run_id, keypoints)
        csv_url = url_for("static", filename="results/" + csv_filename)

        return jsonify(
            success=True,
            preview_image_url=url_for("static", filename="results/" + img_name),
            csv_url=csv_url,
        )
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500


if __name__ == "__main__":
    app.run(debug=True)
