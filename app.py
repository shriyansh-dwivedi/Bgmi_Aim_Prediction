import streamlit as st
import cv2
import tempfile
import math
from ultralytics import YOLO

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="BGMI Aim Analyzer",
    page_icon="🎯",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.main-title {
    font-size: 40px;
    font-weight: bold;
    color: #00ffcc;
}
.sub-text {
    color: #9aa0a6;
}
.metric-box {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<p class="main-title">🎯 BGMI Aim Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Analyze your aim accuracy using AI</p>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Settings")

threshold = st.sidebar.slider("Aim Sensitivity (distance)", 50, 200, 100)
show_boxes = st.sidebar.checkbox("Show Bounding Boxes", True)

uploaded_file = st.sidebar.file_uploader("📤 Upload Gameplay", type=["mp4"])

# ------------------ MAIN LAYOUT ------------------
col1, col2 = st.columns([3,1])

with col1:
    st.subheader("📺 Live Analysis")

    frame_placeholder = st.empty()

with col2:
    st.subheader("📊 Stats")

    good_metric = st.empty()
    bad_metric = st.empty()
    accuracy_metric = st.empty()

# ------------------ PROCESS ------------------
if uploaded_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    model = YOLO("yolov8n.pt")

    good_frames = 0
    total_frames = 0

    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        total_frames += 1

        h, w, _ = frame.shape

        crosshair_x, crosshair_y = w // 2, h // 2
        cv2.circle(frame, (crosshair_x, crosshair_y), 6, (0,0,255), -1)

        results = model(frame)

        frame_good = False

        for box in results[0].boxes:
            cls = int(box.cls[0])

            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                enemy_x = (x1 + x2) // 2
                enemy_y = (y1 + y2) // 2

                cv2.circle(frame, (enemy_x, enemy_y), 6, (255,0,0), -1)

                distance = math.sqrt(
                    (enemy_x - crosshair_x)**2 +
                    (enemy_y - crosshair_y)**2
                )

                if distance < threshold:
                    text = "GOOD AIM"
                    color = (0,255,0)
                    frame_good = True
                else:
                    text = "BAD AIM"
                    color = (0,0,255)

                if show_boxes:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if frame_good:
            good_frames += 1

        # ------------------ UI UPDATE ------------------
        accuracy = (good_frames / total_frames) * 100

        good_metric.metric("✅ Good Frames", good_frames)
        bad_metric.metric("❌ Bad Frames", total_frames - good_frames)
        accuracy_metric.metric("🎯 Accuracy %", f"{accuracy:.2f}")

        frame_placeholder.image(frame, channels="BGR")

        progress_bar.progress(min(total_frames / 300, 1.0))  # fake progress

    cap.release()

    st.success("✅ Analysis Completed!")
