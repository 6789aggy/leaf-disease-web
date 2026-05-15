from flask import Flask, render_template, Response, request, redirect, send_file, url_for
import cv2
import os
import pandas as pd
from datetime import datetime
from ultralytics import YOLO

app = Flask(__name__)

MODEL_PATH = "best.pt"
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
CAPTURE_FOLDER = "static/capture"
CSV_FILE = "history.csv"
CAMERA_INDEX = 0
AUTO_CAPTURE_INTERVAL = 5
HISTORY_INTERVAL = 3

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

VALID_DISEASE_CLASSES = {
    "Anthracnose",
    "Bacterial_Blight",
    "Deficiency Leaf",
    "DryLeaf_Lemon",
    "Dry_Leaf-Lychee",
    "Powdery_Mildew",
    "downey_mildew"
}

disease_map = {
    "Anthracnose": {
        "ascii": "Benh than thu",
        "display": "Bệnh thán thư",
        "severity": "CAO",
        "health": 35,
        "advice": "Cần xử lý nấm bệnh sớm."
    },
    "Bacterial_Blight": {
        "ascii": "Chay la vi khuan",
        "display": "Cháy lá vi khuẩn",
        "severity": "CAO",
        "health": 30,
        "advice": "Kiểm tra độ ẩm và xử lý vi khuẩn."
    },
    "Deficiency Leaf": {
        "ascii": "Thieu dinh duong",
        "display": "Thiếu dinh dưỡng",
        "severity": "TRUNG BÌNH",
        "health": 70,
        "advice": "Bổ sung dinh dưỡng."
    },
    "DryLeaf_Lemon": {
        "ascii": "La kho",
        "display": "Lá khô",
        "severity": "TRUNG BÌNH",
        "health": 65,
        "advice": "Kiểm tra tưới tiêu."
    },
    "Dry_Leaf-Lychee": {
        "ascii": "La kho stress",
        "display": "Lá khô stress",
        "severity": "TRUNG BÌNH",
        "health": 60,
        "advice": "Cây bị stress môi trường."
    },
    "Powdery_Mildew": {
        "ascii": "Nam phan trang",
        "display": "Nấm phấn trắng",
        "severity": "CAO",
        "health": 40,
        "advice": "Giảm độ ẩm."
    },
    "downey_mildew": {
        "ascii": "Benh suong mai",
        "display": "Bệnh sương mai",
        "severity": "CAO",
        "health": 25,
        "advice": "Xử lý sớm."
    }
}

last_auto_capture = 0
last_history_save = 0

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "time",
        "source",
        "disease",
        "confidence",
        "severity",
        "health"
    ]).to_csv(CSV_FILE, index=False)


def sanitize_filename(name):
    safe = ""
    for c in name:
        if c.isalnum() or c in ["_", "-"]:
            safe += c
        else:
            safe += "_"
    return safe


def save_history(source, disease, confidence, severity, health):
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "source": source,
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
        "health": health
    }])

    row.to_csv(
        CSV_FILE,
        mode="a",
        header=False,
        index=False
    )


def preprocess_frame(frame):
    h, w = frame.shape[:2]

    scale = 640 / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    frame = cv2.resize(frame, (new_w, new_h))

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return frame


def detect_frame(frame, source="Camera", allow_auto=True, save_hist=True):
    global last_auto_capture
    global last_history_save

    frame = preprocess_frame(frame)

    results = model(frame, verbose=False)
    result = results[0]

    summary = {
        "detected": False,
        "disease": "Không phát hiện",
        "confidence": 0,
        "severity": "BÌNH THƯỜNG",
        "health": 100,
        "advice": "Không phát hiện bệnh"
    }

    if result.boxes is None or len(result.boxes) == 0:
        return frame, summary

    best_box = None
    best_conf = 0

    for box in result.boxes:
        conf = float(box.conf[0])
        cls = model.names[int(box.cls[0])]

        if cls not in VALID_DISEASE_CLASSES:
            continue

        if conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return frame, summary

    conf = float(best_box.conf[0])

    if conf < 0.20:
        return frame, summary

    cls = model.names[int(best_box.cls[0])]
    info = disease_map[cls]

    ascii_name = info["ascii"]
    display_name = info["display"]
    severity = info["severity"]
    health = info["health"]
    advice = info["advice"]

    confidence = round(conf * 100, 2)

    x1, y1, x2, y2 = map(int, best_box.xyxy[0])

    color = (0, 0, 255) if severity == "CAO" else (0, 255, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    cv2.putText(
        frame,
        ascii_name,
        (x1, max(30, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    summary = {
        "detected": True,
        "disease": display_name,
        "confidence": confidence,
        "severity": severity,
        "health": health,
        "advice": advice
    }

    now = datetime.now().timestamp()

    if save_hist and now - last_history_save >= HISTORY_INTERVAL:
        save_history(
            source,
            display_name,
            confidence,
            severity,
            health
        )
        last_history_save = now

    if allow_auto and severity == "CAO":
        if now - last_auto_capture >= AUTO_CAPTURE_INTERVAL:
            filename = (
                sanitize_filename(display_name)
                + "_auto_"
                + datetime.now().strftime("%Y%m%d_%H%M%S")
                + ".jpg"
            )

            cv2.imwrite(
                os.path.join(CAPTURE_FOLDER, filename),
                frame
            )

            last_auto_capture = now

    return frame, summary


def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        return

    while True:
        success, frame = cap.read()

        if not success:
            break

        frame, _ = detect_frame(
            frame,
            "Camera",
            True,
            True
        )

        ok, buffer = cv2.imencode(".jpg", frame)

        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )

    cap.release()


@app.route("/")
def index():
    df = pd.read_csv(CSV_FILE)

    total = len(df)
    high = 0
    top_disease = "Chưa có"

    if total > 0:
        high = len(df[df["severity"] == "CAO"])
        top_disease = df["disease"].value_counts().index[0]

    return render_template(
        "index.html",
        total=total,
        high=high,
        top_disease=top_disease
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/capture_manual")
def capture_manual():
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        return redirect(url_for("index"))

    success, frame = cap.read()
    cap.release()

    if success:
        frame, result = detect_frame(
            frame,
            "Chụp thủ công",
            False,
            True
        )

        filename = (
            sanitize_filename(result["disease"])
            + "_manual_"
            + datetime.now().strftime("%Y%m%d_%H%M%S")
            + ".jpg"
        )

        cv2.imwrite(
            os.path.join(CAPTURE_FOLDER, filename),
            frame
        )

    return redirect(url_for("index"))


@app.route("/upload_image", methods=["GET", "POST"])
def upload_image():
    result = None
    image_url = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename:
            ext = os.path.splitext(file.filename)[1]
            filename = datetime.now().strftime("%Y%m%d%H%M%S") + ext

            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            frame = cv2.imread(filepath)

            if frame is not None:
                frame, result = detect_frame(
                    frame,
                    "Ảnh",
                    False,
                    True
                )

                output_filename = "detected_" + filename
                output_path = os.path.join(
                    OUTPUT_FOLDER,
                    output_filename
                )

                cv2.imwrite(output_path, frame)

                image_url = "/static/outputs/" + output_filename

    return render_template(
        "upload_image.html",
        result=result,
        image_url=image_url
    )


@app.route("/upload_video", methods=["GET", "POST"])
def upload_video():
    output_video = None

    if request.method == "POST":
        file = request.files.get("video")

        if file and file.filename:
            filename = datetime.now().strftime("%Y%m%d%H%M%S_") + sanitize_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(filepath)

            cap = cv2.VideoCapture(filepath)

            fps = cap.get(cv2.CAP_PROP_FPS) or 20
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

            output_path = os.path.join(
                OUTPUT_FOLDER,
                "detected_" + filename + ".mp4"
            )

            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            while True:
                success, frame = cap.read()

                if not success:
                    break

                processed, _ = detect_frame(
                    frame,
                    "Video",
                    False,
                    False
                )

                processed = cv2.resize(processed, (width, height))
                writer.write(processed)

            cap.release()
            writer.release()

            output_video = output_path.replace("\\", "/")

    return render_template(
        "upload_video.html",
        output_video=output_video
    )


@app.route("/history")
def history():
    df = pd.read_csv(CSV_FILE)
    records = df.to_dict(orient="records")

    return render_template("history.html", records=records)


@app.route("/download_report")
def download_report():
    return send_file(CSV_FILE, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)