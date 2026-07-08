import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ----------------------------------------------------
# Streamlit Page Configuration
# ----------------------------------------------------
st.set_page_config(
    page_title="Animal Detection using YOLO",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Real-Time Animal Detection using YOLO")
st.write("Detect animals from an uploaded image or webcam using the YOLOv8 model.")

# ----------------------------------------------------
# Load YOLO Model
# ----------------------------------------------------
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")
    return model

model = load_model()

# COCO class names
class_names = model.names

# ----------------------------------------------------
# Animal Classes in COCO Dataset
# ----------------------------------------------------
ANIMAL_CLASSES = [
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
]

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("Settings")

option = st.sidebar.radio(
    "Choose Detection Mode",
    [
        "Image Detection",
        "Webcam Detection"
    ]
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=1.00,
    value=0.40,
    step=0.05
)

st.sidebar.write("---")
st.sidebar.write("Animals Supported")

for animal in ANIMAL_CLASSES:
    st.sidebar.write("✅ " + animal.capitalize())

# ----------------------------------------------------
# Main Placeholders
# ----------------------------------------------------
image_placeholder = st.empty()

status_placeholder = st.empty()

info_placeholder = st.empty()

detected_animals_placeholder = st.empty()

confidence_placeholder = st.empty()

# ----------------------------------------------------
# IMAGE DETECTION
# ----------------------------------------------------
if option == "Image Detection":

    uploaded_file = st.file_uploader(
        "Upload an Animal Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # Read image
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        st.subheader("Original Image")
        st.image(image, use_container_width=True)

        # Run YOLO
        results = model(image_np)

        detected_animals = []

        # Process detections
        for result in results:

            boxes = result.boxes

            for box in boxes:

                cls = int(box.cls[0])
                conf = float(box.conf[0])

                class_name = class_names[cls]

                # Detect only animals
                if (
                    class_name in ANIMAL_CLASSES
                    and conf >= confidence_threshold
                ):

                    detected_animals.append(
                        f"{class_name.capitalize()} ({conf:.2f})"
                    )

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # Draw Bounding Box
                    cv2.rectangle(
                        image_np,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    # Draw Label
                    label = f"{class_name} {conf:.2f}"

                    cv2.putText(
                        image_np,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

        st.subheader("Detection Result")
        image_placeholder.image(
            image_np,
            use_container_width=True
        )

        # Show Results
        if len(detected_animals) > 0:

            status_placeholder.success(
                "✅ Animal Detected"
            )

            detected_animals_placeholder.write(
                "### Detected Animals"
            )

            for animal in detected_animals:
                st.write("🐾", animal)

            confidence_placeholder.success(
                f"Total Animals: {len(detected_animals)}"
            )

        else:

            status_placeholder.error(
                "❌ No Animal Detected"
            )

            detected_animals_placeholder.empty()

            confidence_placeholder.empty()

            # ----------------------------------------------------
# WEBCAM DETECTION
# ----------------------------------------------------
if option == "Webcam Detection":

    st.subheader("📷 Real-Time Animal Detection")

    start_camera = st.button("▶ Start Camera")
    stop_camera = st.button("⏹ Stop Camera")

    # Session state
    if "camera_running" not in st.session_state:
        st.session_state.camera_running = False

    if start_camera:
        st.session_state.camera_running = True

    if stop_camera:
        st.session_state.camera_running = False

    video_placeholder = st.empty()

    if st.session_state.camera_running:

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Unable to access webcam.")
            st.stop()

        while st.session_state.camera_running:

            ret, frame = cap.read()

            if not ret:
                st.error("Failed to read frame.")
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Display original webcam feed
            video_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

            status_placeholder.info(
                "Camera Running..."
            )

        cap.release()

        cv2.destroyAllWindows()

        while st.session_state.camera_running:

         ret, frame = cap.read()

    if not ret:
        st.error("Failed to read frame")

    # -------------------------------
    # Run YOLO on webcam frame
    # -------------------------------
    results = model(frame)

    detected_animals = []

    # Process all detections
    for result in results:

        for box in result.boxes:

            cls = int(box.cls[0])
            conf = float(box.conf[0])

            class_name = class_names[cls]

            # Detect only animals
            if (
                class_name in ANIMAL_CLASSES
                and conf >= confidence_threshold
            ):

                detected_animals.append(
                    f"{class_name.capitalize()} ({conf:.2f})"
                )

                # Bounding Box
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Green Rectangle
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                label = f"{class_name} {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # -------------------------------
    # Convert to RGB
    # -------------------------------
    frame_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    video_placeholder.image(
        frame_rgb,
        channels="RGB",
        use_container_width=True
    )

    # -------------------------------
    # Display Detection Results
    # -------------------------------
    if len(detected_animals) > 0:

        status_placeholder.success(
            "✅ Animal Detected"
        )

        detected_animals_placeholder.write(
            "### Detected Animals"
        )

        for animal in detected_animals:
            st.write("🐾", animal)

        confidence_placeholder.success(
            f"Animals Detected: {len(detected_animals)}"
        )

    else:

        status_placeholder.warning(
            "❌ No Animal Detected"
        )

        detected_animals_placeholder.empty()

        confidence_placeholder.empty()