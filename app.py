from flask import Flask, render_template, request, jsonify
from src.utils import get_dist
import numpy as np
app = Flask(__name__)
    


@app.route("/")
def index():
    return render_template("page.html")

@app.route("/pixel", methods=["POST"])
def pixel_data():
    data = request.get_json()
    x1 = data.get("x1")
    y1 = data.get("y1")
    x2 = data.get("x2")
    y2 = data.get("y2")
    color = data.get("color")  # optional
    # print(f"Received pixel: ({x}, {y}) color={color}")
    pt1 = np.array([x1,y1])
    pt2 = np.array([x2,y2])

    dist = get_dist(pt1,pt2)
    # You can process or log this on the backend

    return jsonify({"status": "success", "message": f"Distance between points: {dist:.2f} in"})

if __name__ == "__main__":
    app.run(debug=True)
