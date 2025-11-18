# image_traffic_app.py
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# Try to import ultralytics YOLO; if not installed, show instructions
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False

# Try to load ML components if they exist
use_ml_model = False
try:
    import joblib
    if os.path.exists("rf_model.joblib") and os.path.exists("scaler.joblib") and os.path.exists("label_encoder.joblib"):
        rf_model = joblib.load("rf_model.joblib")
        scaler = joblib.load("scaler.joblib")
        label_encoder = joblib.load("label_encoder.joblib")
        use_ml_model = True
except Exception:
    use_ml_model = False

st.set_page_config(page_title="Image-based Traffic Manager", layout="centered")
st.title("🚦 Image-Based AI Traffic Management (Eco-Friendly)")

st.markdown(
    """
Upload a traffic image (or use the sample) — the app will detect vehicles using YOLO,
count them, compute density, predict congestion, and recommend eco-friendly signal timings.
"""
)

# Sidebar settings
st.sidebar.header("Settings & Inputs")
lane_capacity = st.sidebar.number_input("Lane capacity (vehicles)", min_value=1, value=100)
avg_speed_input = st.sidebar.slider("If available, set average speed (km/h)", 0, 120, 40)
use_webcam = st.sidebar.checkbox("Use webcam (experimental)", value=False)

# Show YOLO installation note if not available
if not YOLO_AVAILABLE:
    st.warning(
        "YOLO (ultralytics) not installed. To enable automatic vehicle detection, run:\n\n"
        "`pip install ultralytics opencv-python-headless` \n\n"
        "Or continue and use the fallback rule-based classifier (still supports image upload)."
    )

# Load or create YOLO model (this will auto-download if needed)
yolo_model = None
if YOLO_AVAILABLE:
    with st.spinner("Loading YOLO model..."):
        try:
            yolo_model = YOLO("yolov8n.pt")  # small model; will auto-download if missing
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            yolo_model = None

# Helper functions
VEHICLE_COCO_IDS = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck (COCO class ids)

def detect_vehicles_yolo(image_bgr):
    """
    Runs YOLO detection and returns:
    - annotated_bgr (image with boxes)
    - vehicle_count (int)
    """
    if yolo_model is None:
        return image_bgr, 0

    # Ultraytics YOLO expects RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb)[0]  # first/only result

    annotated = image_bgr.copy()
    vehicle_count = 0

    # results.boxes may be empty
    try:
        for box in results.boxes:
            cls_id = int(box.cls[0]) if hasattr(box, "cls") else int(box.cls)
            if cls_id in VEHICLE_COCO_IDS:
                vehicle_count += 1
                # box.xyxy is tensor-like -> convert to ints
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0]) if hasattr(box, "conf") else float(box.conf)
                label = f"{cls_id}:{conf:.2f}"
                # Draw rectangle and label
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
                cv2.putText(annotated, f"{conf:.2f}", (x1, max(y1-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    except Exception:
        # Fallback if structure differs
        pass

    return annotated, vehicle_count

def rule_based_congestion(vehicle_count, avg_speed, lane_capacity):
    density = vehicle_count / lane_capacity
    # Simple heuristic: density + speed threshold
    if density >= 0.7 or avg_speed <= 25:
        return "High"
    elif density >= 0.35 or avg_speed <= 45:
        return "Medium"
    else:
        return "Low"

def predict_congestion(vehicle_count, avg_speed, lane_capacity):
    density = vehicle_count / lane_capacity
    X = np.array([[vehicle_count, avg_speed, density]])
    if use_ml_model:
        try:
            X_scaled = scaler.transform(X)
            y_pred = rf_model.predict(X_scaled)
            label = label_encoder.inverse_transform(y_pred)[0]
            return label, density
        except Exception:
            # fallback
            return rule_based_congestion(vehicle_count, avg_speed, lane_capacity), density
    else:
        return rule_based_congestion(vehicle_count, avg_speed, lane_capacity), density

def optimize_signal(congestion_level):
    if congestion_level == "High":
        return {"Green Time": "60s", "Red Time": "30s"}
    elif congestion_level == "Medium":
        return {"Green Time": "45s", "Red Time": "45s"}
    else:
        return {"Green Time": "30s", "Red Time": "60s"}

# Image input section
col1, col2 = st.columns([1, 1])
with col1:
    uploaded_file = st.file_uploader("Upload traffic image", type=["jpg", "jpeg", "png"])
    if st.button("Use sample image"):
        uploaded_file = "sample"

with col2:
    st.write("Or adjust inputs:")
    st.write(f"Lane capacity = **{lane_capacity}**")
    st.write(f"Average speed (for model) = **{avg_speed_input} km/h**")

# Process webcam if requested
if use_webcam and YOLO_AVAILABLE:
    st.info("Webcam mode: click 'Start Webcam' and allow camera permission in the browser.")
    run_cam = st.button("Start Webcam")
    if run_cam:
        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Cannot open webcam")
        else:
            stop = st.button("Stop Webcam")
            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated, count = detect_vehicles_yolo(frame)
                congestion_label, density = predict_congestion(count, avg_speed_input, lane_capacity)
                signal = optimize_signal(congestion_label)

                # Overlay simple text
                cv2.putText(annotated, f"Vehicles: {count}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(annotated, f"Congestion: {congestion_label}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                FRAME_WINDOW.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            cap.release()

# Normal image flow
if uploaded_file:
    if uploaded_file == "sample":
        # load sample image packaged with repo if exists
        sample_path = "sample_traffic.jpg"
        if os.path.exists(sample_path):
            pil_img = Image.open(sample_path).convert("RGB")
            image_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            st.error("No sample image found in project folder (place sample_traffic.jpg).")
            st.stop()
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Original image", use_column_width=True)

    # Detection
    if YOLO_AVAILABLE and yolo_model is not None:
        with st.spinner("Detecting vehicles..."):
            annotated_img, vehicle_count = detect_vehicles_yolo(image_bgr)
    else:
        annotated_img = image_bgr.copy()
        vehicle_count = 0
        st.info("YOLO not available — automatic detection skipped. You can still input vehicle_count manually.")
        manual_count = st.number_input("Or enter detected vehicle count manually", min_value=0, value=0)
        vehicle_count = manual_count

    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Detected vehicles", use_column_width=True)
    st.write(f"**Detected Vehicles:** {vehicle_count}")

    # Predict congestion
    congestion_label, density = predict_congestion(vehicle_count, avg_speed_input, lane_capacity)
    st.write(f"**Density:** {density:.3f}")
    st.subheader(f"Predicted Congestion Level: {congestion_label}")

    # Signal suggestion
    signal = optimize_signal(congestion_label)
    st.write("### Suggested Signal Timings")
    st.json(signal)

    # Optionally save detection result
    if st.button("Save annotated image"):
        save_path = "annotated_result.jpg"
        cv2.imwrite(save_path, annotated_img)
        st.success(f"Annotated image saved to {save_path}")

else:
    st.info("Upload an image to begin detection.")
