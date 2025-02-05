from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import logging
from detector import ObjectDetector
from descriptions import get_description

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
detector = ObjectDetector()


def process_detections(results, img):
    """Process detection results with enhanced information"""
    try:
        processed = []
        for result in results:
            box = result["box"]
            # Pull color, size, and position from [`detector.py`](detector.py)
            color = result.get("color", detector.get_dominant_color(img, box))
            size = result.get("size", "unknown")
            position = result.get("position", "unknown")

            processed.append(
                {
                    "class_id": int(result["class_id"]),
                    "class_name": result["class_name"],
                    "confidence": float(result["confidence"]) * 100,
                    "box": box.tolist(),
                    "color": color,
                    "size": size,
                    "position": position,
                    "description": get_description(result["class_name"], color),
                }
            )
        return processed
    except Exception as e:
        logger.error(f"Error processing detections: {str(e)}")
        raise


def draw_detection(image, result):
    """Draw detection box and label on image with dynamic sizing"""
    try:
        box = result["box"]
        color = result.get("color", "unknown")
        size = result.get("size", "unknown")
        position = result.get("position", "unknown")
        confidence = result["confidence"]

        # Calculate dynamic font size and thickness based on image size
        height, width = image.shape[:2]
        base_font_scale = min(width, height) / 1000.0

        # Increase font size for high confidence
        if confidence > 90:
            font_scale = base_font_scale * 1.5
            thickness = max(2, int(base_font_scale * 3))
        else:
            font_scale = base_font_scale
            thickness = max(1, int(base_font_scale * 2))

        label = (
            f"{result['class_name']} ({color}, {size}, {position}): {confidence:.1f}%"
        )

        # Draw bounding box
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0),
            thickness,
        )

        # Draw label
        label_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )[0]
        top = max(int(box[1]), label_size[1])
        cv2.rectangle(
            image,
            (int(box[0]), top - label_size[1] - 5),
            (int(box[0]) + label_size[0], top + 5),
            (0, 255, 0),
            cv2.FILLED,
        )
        cv2.putText(
            image,
            label,
            (int(box[0]), top),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
        )
        return image
    except Exception as e:
        logger.error(f"Error drawing detection: {str(e)}")
        return image


@app.route("/favicon.ico")
def favicon():
    return app.send_static_file("favicon.ico")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        file = request.files["image"]
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"success": False, "error": "Invalid image format"}), 400

        # Run detection from [`detector.ObjectDetector`](detector.py)
        results = detector.detect(img)
        processed_results = process_detections(results, img)

        # Draw and encode results
        for result in processed_results:
            img = draw_detection(img, result)
        _, buffer = cv2.imencode(".jpg", img)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "count": len(processed_results),
                "results": processed_results,
                "image": img_base64,
            }
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
