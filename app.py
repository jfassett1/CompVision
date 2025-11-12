from flask import Flask, render_template, request, jsonify
from src.utils import get_dist
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("page.html")


@app.route("/2")
def hw2():
    return render_template("hw2.html")


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


if __name__ == "__main__":
    app.run(debug=True)
