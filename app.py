import streamlit as st
import os
import json
import urllib.request
import pandas as pd
import numpy as np
import cv2
import onnxruntime as ort
import plotly.express as px
import streamlit.components.v1 as components

# Declare custom component for client-side WebRTC inference (Scenario A)
parent_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(parent_dir, "frontend")
stress_webrtc_component = components.declare_component("stress_webrtc", path=frontend_dir)
import threading
import time
import csv
import datetime
import tempfile

# Set Page Config
st.set_page_config(page_title="Student Stress Monitor", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Resolve paths dynamically
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "output", "emotion_stress_cnn.onnx")
CASCADE_PATH = os.path.join(current_dir, "Pack01b_DeploymentCode", "haarcascade_frontalface_default.xml")
LOG_PATH = os.path.join(current_dir, "output", "stress_log.csv")

# Load Model & Cascade (Cached to avoid memory leaks)
@st.cache_resource
def load_resources():
    session = ort.InferenceSession(MODEL_PATH)
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return session, cascade

try:
    session, face_cascade = load_resources()
    input_name = session.get_inputs()[0].name
except Exception as e:
    st.error(f"Failed to load resources: {e}. Please ensure the model file is in 'output/' and the haarcascade is in 'Pack01b_DeploymentCode/'.")
    st.stop()

labels_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}



# Thread-safe state for the background video processor
class StressState:
    def __init__(self):
        self.stress_counter = 0
        self.last_alert_time = 0
        self.lock = threading.Lock()

@st.cache_resource
def get_state():
    return StressState()

state = get_state()

# Cloud stress state (persists across reruns for camera-input mode)
if "cloud_stress" not in st.session_state:
    st.session_state.cloud_stress = {"counter": 0, "last_alert": 0.0}



# Session state navigation management
if "active_page" not in st.session_state:
    st.session_state.active_page = "Stress Detection Portal"

def on_portal_change():
    st.session_state.active_page = st.session_state.nav_portal

def on_analytics_change():
    st.session_state.active_page = st.session_state.nav_analytics

page = st.session_state.active_page

# Header Layout
if page == "Stress Detection Portal":
    st.markdown("### 🧠 Student Stress Detection <span style='font-size: 14px; font-weight: normal; color: #a3a8b4; margin-left: 10px;'>| Real-time Emotion & Stress Tracking</span>", unsafe_allow_html=True)
else:
    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 📊 Student Stress Analytics Dashboard")
    with header_col2:
        page = st.selectbox(
            "Navigation", 
            ["Stress Detection Portal", "Analytics Dashboard"], 
            key="nav_analytics",
            index=1,
            on_change=on_analytics_change,
            label_visibility="collapsed"
        )

# Thin custom divider to save vertical space
st.markdown("<div style='border-bottom: 1px solid #30363d; margin-top: -10px; margin-bottom: 15px;'></div>", unsafe_allow_html=True)

if page == "Stress Detection Portal":
    # Context-aware environment detection
    is_streamlit_cloud = os.path.exists("/mount/src")
    is_hugging_face = "SPACE_ID" in os.environ
    is_cloud = is_streamlit_cloud or is_hugging_face
    
    mode = "🎥 Real-time Webcam"
    
    # Split layout 1:1 to keep columns balanced at 50:50
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if mode == "🎥 Real-time Webcam":
            if is_cloud:
                # Render client-side WebRTC inference component (Scenario A)
                telemetry = stress_webrtc_component(key="cloud_webrtc")
                
                if telemetry:
                    stress_val = telemetry.get("stress_level", 0)
                    emotion = telemetry.get("emotion", "Neutral")
                    confidence = telemetry.get("confidence", 0.0)
                    
                    st.session_state.cloud_stress["counter"] = stress_val
                    
                    # Logging alert to CSV if stress is high (throttled to once every 2 seconds)
                    if stress_val >= 30:
                        current_time = time.time()
                        if current_time - st.session_state.cloud_stress["last_alert"] > 2.0:
                            st.session_state.cloud_stress["last_alert"] = current_time
                            timestamp = telemetry.get("client_time", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                            file_exists = os.path.exists(LOG_PATH)
                            with open(LOG_PATH, "a", newline="") as f:
                                writer = csv.writer(f)
                                if not file_exists or os.path.getsize(LOG_PATH) == 0:
                                    writer.writerow(["Timestamp", "Stress Level"])
                                writer.writerow([timestamp, stress_val])

                else:
                    st.warning("⚠️ For local deployment, please run 'appLocal.py' instead to use direct local WebRTC.")
        else:
            st.markdown("#### 📷 Photo Source")
            if is_cloud:
                photo_source = "📁 Upload from File Explorer"
            else:
                photo_source = st.radio(
                    "Choose photo source", 
                    ["📁 Upload from File Explorer", "📸 Take a Camera Snapshot"], 
                    horizontal=True,
                    label_visibility="collapsed"
                )
            
            uploaded_file = None
            if photo_source == "📸 Take a Camera Snapshot":
                uploaded_file = st.camera_input("Capture student face snapshot")
            else:
                uploaded_file = st.file_uploader(
                    "Upload a photo or a short video (max 10s)...", 
                    type=["jpg", "jpeg", "png", "mp4", "mov", "avi"]
                )
                
            if uploaded_file is not None:
                file_name = uploaded_file.name.lower()
                is_video = file_name.endswith((".mp4", ".mov", ".avi"))
                
                if is_video:
                    st.write("🔄 **Analyzing video frames...**")
                    # Save video locally to read with OpenCV
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tfile.write(uploaded_file.read())
                    tfile.close()
                    
                    cap = cv2.VideoCapture(tfile.name)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if fps <= 0:
                        fps = 30.0
                        
                    frame_count = 0
                    stress_timeline = []
                    timestamps = []
                    emotions_list = []
                    
                    # Create progress visual indicators
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Process at max 6 frames per second (skip frames to avoid cloud timeouts)
                    frame_skip = max(1, int(fps / 6))
                    max_frames = int(fps * 10)  # Max 10 seconds of video
                    current_stress = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret or frame_count > max_frames:
                            break
                            
                        if frame_count % frame_skip == 0:
                            # Update progress
                            pct = min(100, int((frame_count / max_frames) * 100))
                            progress_bar.progress(pct)
                            
                            # Convert to Grayscale
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                            
                            frame_stress_added = 0
                            for (x, y, w, h) in faces:
                                roi = gray[y:y+h, x:x+w]
                                roi = cv2.resize(roi, (48, 48)) / 255.0
                                roi = roi.reshape(1, 48, 48, 1).astype(np.float32)
                                
                                predictions = session.run(None, {input_name: roi})[0][0]
                                class_id = np.argmax(predictions)
                                emotions_list.append(labels_dict[class_id])
                                
                                if class_id in [0, 2]:  # Angry, Fear
                                    frame_stress_added += 25
                                else:
                                    frame_stress_added -= 5
                                    
                            if len(faces) > 0:
                                current_stress = min(100, max(0, current_stress + frame_stress_added))
                            else:
                                current_stress = max(0, current_stress - 3)
                                
                            stress_timeline.append(current_stress)
                            timestamps.append(frame_count / fps)
                            
                        frame_count += 1
                        
                    cap.release()
                    try:
                        os.unlink(tfile.name)
                    except Exception:
                        pass
                        
                    progress_bar.empty()
                    
                    if len(stress_timeline) > 0:
                        avg_stress = float(np.mean(stress_timeline))
                        peak_stress = int(np.max(stress_timeline))
                        
                        # Update global state
                        with state.lock:
                            state.stress_counter = int(stress_timeline[-1])
                            
                            # Write log entries to CSV for telemetry/analytics
                            for ts, val in zip(timestamps, stress_timeline):
                                if val >= 30:
                                    log_time = (datetime.datetime.now() - datetime.timedelta(seconds=(timestamps[-1] - ts))).strftime("%Y-%m-%d %H:%M:%S")
                                    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                                    file_exists = os.path.exists(LOG_PATH)
                                    with open(LOG_PATH, "a", newline="") as f:
                                        writer = csv.writer(f)
                                        if not file_exists or os.path.getsize(LOG_PATH) == 0:
                                            writer.writerow(["Timestamp", "Stress Level"])
                                        writer.writerow([log_time, val])
                                        
                        st.success("✅ **Video Analysis Completed!**")
                        
                        # Plot line chart
                        df_video = pd.DataFrame({
                            "Time (Seconds)": timestamps,
                            "Stress Level (%)": stress_timeline
                        })
                        fig = px.line(df_video, x="Time (Seconds)", y="Stress Level (%)", title="Stress Index Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Report metrics
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Average Stress", f"{avg_stress:.1f}%")
                        m2.metric("Peak Stress", f"{peak_stress}%")
                        if emotions_list:
                            most_common = max(set(emotions_list), key=emotions_list.count)
                            m3.metric("Dominant Emotion", most_common)
                    else:
                        st.error("No valid frames could be processed from the video.")
                        
                else:
                    # Process the static image
                    file_bytes = uploaded_file.read()
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                        
                        stress_added = 0
                        detected_emotions = []
                        
                        for (x, y, w, h) in faces:
                            roi = gray[y:y+h, x:x+w]
                            roi = cv2.resize(roi, (48, 48)) / 255.0
                            roi = roi.reshape(1, 48, 48, 1).astype(np.float32)
                            
                            predictions = session.run(None, {input_name: roi})[0][0]
                            class_id = np.argmax(predictions)
                            conf = predictions[class_id]
                            
                            detected_emotions.append(labels_dict[class_id])
                            
                            # Stress logic (Angry=0, Fear=2)
                            if class_id in [0, 2]:
                                stress_added = min(100, stress_added + 35)
                            else:
                                stress_added = max(0, stress_added + 5)
                                
                            color = (0, 0, 255) if class_id in [0, 2] else (0, 255, 0)
                            cv2.rectangle(img, (x, y), (x+w, y+h), color, 4)
                            # Text annotation
                            cv2.putText(img, f"{labels_dict[class_id]} ({conf*100:.0f}%)", (x, y - 12), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
                        
                        # Update stress state globally
                        with state.lock:
                            if len(faces) > 0:
                                state.stress_counter = min(100, state.stress_counter + stress_added)
                                if stress_added > 0:
                                    state.stress_counter = min(100, max(30, state.stress_counter))
                                else:
                                    state.stress_counter = max(0, state.stress_counter - 10)
                            else:
                                state.stress_counter = max(0, state.stress_counter - 5)
                            
                            curr_stress = state.stress_counter
                            
                            # Log alert to CSV if stress is high
                            if curr_stress >= 30 and len(faces) > 0:
                                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                                file_exists = os.path.exists(LOG_PATH)
                                with open(LOG_PATH, "a", newline="") as f:
                                    writer = csv.writer(f)
                                    if not file_exists or os.path.getsize(LOG_PATH) == 0:
                                        writer.writerow(["Timestamp", "Stress Level"])
                                    writer.writerow([timestamp, curr_stress])
                        
                        # Display processed image (Convert BGR to RGB for Streamlit)
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        st.image(img_rgb, caption="Processed Image", use_container_width=True)
                        
                        if len(faces) > 0:
                            st.write(f"Detected Emotions: **{', '.join(detected_emotions)}**")
                        else:
                            st.write("No faces detected in the image.")
                    else:
                        st.error("Failed to decode uploaded image. Please try another file.")
    
    with col2:
        # Split col2 header to put navigation selectbox next to "Status & Telemetry" on desktop
        status_header_col1, status_header_col2 = st.columns([2, 1])
        with status_header_col1:
            st.markdown("##### 📊 Status & Telemetry")
        with status_header_col2:
            page = st.selectbox(
                "Navigation", 
                ["Stress Detection Portal", "Analytics Dashboard"], 
                key="nav_portal",
                index=0,
                on_change=on_portal_change,
                label_visibility="collapsed"
            )
        
        # Real-time dashboard feedback
        if is_cloud:
            curr_stress = st.session_state.cloud_stress["counter"]
        else:
            with state.lock:
                curr_stress = state.stress_counter
            
        st.metric("Last Tracked Stress Level", f"{curr_stress}%")
        
        if curr_stress >= 30:
            st.warning("⚠️ **High Stress Detected!** Please take a deep breath.")
        else:
            st.success("✅ **Calm / Normal State**")
            
        st.info(
            "💡 **Detection Guidelines:**\n"
            "- Face box turns **Red** when stress emotions (Angry/Fear) are detected.\n"
            "- Face box turns **Green** when other emotions (Calm) are detected.\n"
            "- Stress level accumulates if stress expressions are consistently tracked."
        )

elif page == "Analytics Dashboard":
    pass
    
    if os.path.exists(LOG_PATH) and os.path.getsize(LOG_PATH) > 20:
        df = pd.read_csv(LOG_PATH)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        
        # Time range selection (Configure visual window)
        st.markdown("#### ⚙️ Configure View")
        time_filter = st.selectbox(
            "Select Time Window", 
            ["All History", "Last 1 Hour", "Last 30 Minutes", "Last 15 Minutes", "Last 10 Minutes", "Last 5 Minutes"]
        )
        
        now = df["Timestamp"].max()
        if pd.isna(now):
            now = pd.Timestamp.now()

        if time_filter == "Last 1 Hour":
            df = df[df["Timestamp"] >= now - pd.Timedelta(hours=1)]
        elif time_filter == "Last 30 Minutes":
            df = df[df["Timestamp"] >= now - pd.Timedelta(minutes=30)]
        elif time_filter == "Last 15 Minutes":
            df = df[df["Timestamp"] >= now - pd.Timedelta(minutes=15)]
        elif time_filter == "Last 10 Minutes":
            df = df[df["Timestamp"] >= now - pd.Timedelta(minutes=10)]
        elif time_filter == "Last 5 Minutes":
            df = df[df["Timestamp"] >= now - pd.Timedelta(minutes=5)]
            
        st.markdown("#### 📈 Stress Level Trend")
        fig = px.line(df, x="Timestamp", y="Stress Level", title="Chronological Stress Level Index")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### 📋 Full Activity Logs")
        st.dataframe(
            df, 
            use_container_width=True,
            column_config={
                "Timestamp": st.column_config.DatetimeColumn(
                    "Timestamp",
                    format="YYYY-MM-DD HH:mm:ss"
                ),
                "Stress Level": st.column_config.NumberColumn(
                    "Stress Level",
                    format="%d%%"
                )
            }
        )

    else:
        st.info("No logs saved yet. Please perform stress detection first.")
