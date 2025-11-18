import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# ---------------------------
# 1. Import YOLO (Ultralytics)
# ---------------------------
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# ---------------------------
# 2. Load ML Models if present
# ---------------------------
use_ml_model = False
try:
    import joblib
    model_path = "models"
    rf_model_file = os.path.join(model_path, "rf_model.joblib")
    scaler_file = os.path.join(model_path, "scaler.joblib")
    encoder_file = os.path.join(model_path, "label_encoder.joblib")

    if os.path.exists(rf_model_file) and os.path.exists(scaler_file) and os.path.exists(encoder_file):
        rf_model = joblib.load(rf_model_file)
        scaler = joblib.load(scaler_file)
        label_encoder = joblib.load(encoder_file)
        use_ml_model = True
except Exception:
    use_ml_model = False


# ---------------------------
# Streamlit UI Setup
# ---------------------------
st.set_page_config(page_title="Image Traffic Manager", layout="wide")
st.title("🚦 Image-Based AI Traffic Management System")

st.write("""
Upload a traffic image — the system will detect vehicles using YOLO, 
predict congestion level, and suggest eco-friendly signal timings.
""")


# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Settings")
lane_capacity = st.sidebar.number_input("Lane Capacity (vehicles)", min_value=10, value=100)
avg_speed_input = st.sidebar.slider("Average Speed (km/h)", 0, 120, 40)


# ---------------------------
# Load YOLO model
# ---------------------------
yolo_model = None
if YOLO_AVAILABLE:
    with st.spinner("Loading YOLOv8 model..."):
        try:
            # YOLO will auto-download yolov8n.pt if missing
            yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            yolo_model = None
else:
    st.warning("YOLO is not installed. Install it using: pip install ultralytics")


# ---------------------------
# Helper: YOLO Vehicle Detection
# ---------------------------
VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bicycle, car, motorcycle, bus, truck

def detect_vehicles(image_bgr):
    if yolo_model is None:
        return image_bgr, 0

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_rgb)[0]

    annotated = image_bgr.copy()
    count = 0

    for box in results.boxes:
        cls = int(box.cls)
        if cls in VEHICLE_CLASSES:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)

    return annotated, count


# ---------------------------
# Congestion Prediction
# ---------------------------
def rule_based(vehicle_count, avg_speed):
    density = vehicle_count / lane_capacity
    if density >= 0.7 or avg_speed <= 25:
        return "High", density
    elif density >= 0.35:
        return "Medium", density
    else:
        return "Low", density

def predict(vehicle_count, avg_speed):
    density = vehicle_count / lane_capacity
    X = np.array([[vehicle_count, avg_speed, density]])

    if use_ml_model:
        try:
            X_scaled = scaler.transform(X)
            pred = rf_model.predict(X_scaled)
            label = label_encoder.inverse_transform(pred)[0]
            return label, density
        except:
            return rule_based(vehicle_count, avg_speed)
    else:
        return rule_based(vehicle_count, avg_speed)


# ---------------------------
# Eco-Friendly Signal Optimization
# ---------------------------
def optimize(level):
    if level == "High":
        return {"Green Time": "60s", "Red Time": "30s"}
    if level == "Medium":
        return {"Green Time": "45s", "Red Time": "45s"}
    return {"Green Time": "30s", "Red Time": "60s"}


# ---------------------------
# Image Upload Section
# ---------------------------
uploaded_file = st.file_uploader("Upload Traffic Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image_bgr, caption="Original Image", use_column_width=True)

    with st.spinner("Detecting vehicles..."):
        annotated_img, vehicle_count = detect_vehicles(image_bgr)

    st.image(annotated_img, caption="Detected Vehicles", use_column_width=True)
    st.write(f"### 🚗 Vehicles Detected: **{vehicle_count}**")

    # Predict congestion
    level, density = predict(vehicle_count, avg_speed_input)
    st.write(f"### Density: **{density:.2f}**")
    st.subheader(f"Predicted Congestion Level: **{level}**")

    # Timing suggestion
    st.write("### Suggested Signal Timings")
    st.json(optimize(level))

else:
    st.info("Upload an image to start the analysis.")
