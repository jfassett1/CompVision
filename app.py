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


if __name__ == "__main__":
    app.run(debug=True)
