import streamlit as st
import os
import json
import urllib.request
import pandas as pd
import numpy as np
import cv2
import onnxruntime as ort
import plotly.express as px
from streamlit_webrtc import webrtc_streamer
import av
import threading
import time
import csv
import datetime
import tempfile

# Set Page Config
st.set_page_config(page_title="Student Stress Monitor", layout="wide")

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

# Cloud RTC configuration with TURN relay (bypasses UDP blocks on Streamlit Cloud / HF Spaces)
def get_cloud_rtc_config():
    """Get TURN-enabled RTC config for cloud environments.
    
    Tier 1: streamlit-webrtc's get_hf_ice_servers() (HF internal infra, DNS resolves).
    Tier 2: Cloudflare TURN via fastrtc endpoint (DNS may fail on HF Spaces).
    Fallback: Metered.ca open relay with TCP transport (universal fallback).
    """
    hf_token = os.environ.get("HF_TOKEN")
    print(f"[RTC] HF_TOKEN present: {bool(hf_token)}")

    # --- DNS Diagnostics ---
    import socket
    for host in [
        "fastrtc-turn-server-login.hf.space",
        "turn.fastrtc.org",
        "openrelay.metered.ca",
    ]:
        try:
            ip = socket.gethostbyname(host)
            print(f"[RTC] DNS OK: {host} -> {ip}")
        except Exception as e:
            print(f"[RTC] DNS FAIL: {host} -> {e}")

    # --- Tier 1: HF native ICE server (internal HF infra, DNS should resolve) ---
    if hf_token:
        try:
            from streamlit_webrtc import get_hf_ice_servers
            raw = get_hf_ice_servers(token=hf_token)
            print(f"[RTC] get_hf_ice_servers() succeeded: {raw}")

            # Modify all TURN URLs to force TCP transport
            ice_servers = []
            for sv in raw:
                urls = sv["urls"]
                if isinstance(urls, str):
                    urls = [urls]
                modified_urls = [
                    u + ("?transport=tcp" if "?" not in u else u)
                    for u in urls
                ]
                ice_servers.append({
                    "urls": modified_urls,
                    "username": sv["username"],
                    "credential": sv["credential"],
                })

            return {
                "iceServers": ice_servers,
                "iceTransportPolicy": "relay",
            }
        except Exception as e:
            print(f"[RTC] HF native ICE servers failed: {e}")
    else:
        print("[RTC] No HF_TOKEN — skipping Tier 1 and Tier 2")

    # --- Tier 2: Cloudflare TURN via fastrtc endpoint ---
    if hf_token:
        try:
            req = urllib.request.Request(
                "https://turn.fastrtc.org/credentials?ttl=600",
                headers={"Authorization": f"Bearer {hf_token}"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                print(f"[RTC] Cloudflare TURN succeeded: {data}")
                return data
        except Exception as e:
            print(f"[RTC] Cloudflare TURN failed: {e}")

    # --- Fallback: Metered.ca open relay over TCP ---
    print("[RTC] Falling back to Metered.ca TURN relay")
    return {
        "iceServers": [
            {
                "urls": [
                    "turn:openrelay.metered.ca:80?transport=tcp",
                    "turn:openrelay.metered.ca:443?transport=tcp",
                    "turns:openrelay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ],
        "iceTransportPolicy": "relay",
    }

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

def process_cloud_frame(image_bytes):
    """Process a single captured frame from st.camera_input on cloud.
    
    Returns (rgb_image_with_HUD, stress_level, emotion_name) or (None, current_stress, None) on error.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return None, st.session_state.cloud_stress["counter"], None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        stress = st.session_state.cloud_stress["counter"]
        detected_emotion = None
        frame_stress_added = 0
        
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (48, 48)) / 255.0
            roi = roi.reshape(1, 48, 48, 1).astype(np.float32)
            
            predictions = session.run(None, {input_name: roi})[0][0]
            class_id = np.argmax(predictions)
            detected_emotion = labels_dict[class_id]
            
            if class_id in [0, 2]:  # Angry, Fear
                frame_stress_added += 25
            else:
                frame_stress_added -= 5
            
            box_color = (0, 0, 255) if class_id in [0, 2] else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, labels_dict[class_id], (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
        
        if len(faces) > 0:
            stress = min(100, max(0, stress + frame_stress_added))
        else:
            stress = max(0, stress - 3)
        
        # Draw stress overlay on frame
        stress_color = (0, 0, 255) if stress > 50 else (0, 255, 0)
        cv2.putText(frame, f"Stress: {stress}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, stress_color, 2)
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame_rgb, stress, detected_emotion
        
    except Exception as e:
        print(f"[CLOUD] Frame processing error: {e}")
        return None, st.session_state.cloud_stress["counter"], None

# Callback for processing video frames
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no face detected, gradually decay stress
    if len(faces) == 0:
        with state.lock:
            state.stress_counter = max(0, state.stress_counter - 1)
            
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48)) / 255.0
        roi = roi.reshape(1, 48, 48, 1).astype(np.float32)
        
        predictions = session.run(None, {input_name: roi})[0][0]
        class_id = np.argmax(predictions)
        conf = predictions[class_id]
        
        with state.lock:
            # Stress Logic (Angry=0 & Fear=2)
            if class_id in [0, 2]:
                state.stress_counter = min(100, state.stress_counter + 2)
            else:
                state.stress_counter = max(0, state.stress_counter - 1)
            
            current_stress = state.stress_counter
            last_alert = state.last_alert_time

        # Draw bounding box (Red for Stressed, Green for Calm/Others)
        color = (0, 0, 255) if class_id in [0, 2] else (0, 255, 0)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        
        # Clean HUD-style text layout above the bounding box
        # 1. Emotion label
        cv2.putText(img, f"{labels_dict[class_id]} ({conf*100:.0f}%)", (x, y - 8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        
        # 2. Progress Bar (6px tall)
        cv2.rectangle(img, (x, y - 26), (x + 100, y - 20), (80, 80, 80), -1)
        cv2.rectangle(img, (x, y - 26), (x + current_stress, y - 20), color, -1)
        
        # 3. Stress level indicator text to the right of the progress bar
        cv2.putText(img, f"Stress: {current_stress}%", (x + 105, y - 19), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 4. Stress Alert label above the meter if stress >= 30
        if current_stress >= 30:
            cv2.putText(img, "ALERT!", (x, y - 34), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Non-blocking log to CSV inside thread
            current_time = time.time()
            if current_time - last_alert > 2.0:
                with state.lock:
                    state.last_alert_time = current_time
                
                # Write to CSV safely
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
                file_exists = os.path.exists(LOG_PATH)
                with open(LOG_PATH, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists or os.path.getsize(LOG_PATH) == 0:
                        writer.writerow(["Timestamp", "Stress Level"])
                    writer.writerow([timestamp, current_stress])
                    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# Header Layout (Title on left, compact navigation dropdown on right)
header_col1, header_col2 = st.columns([3, 1])

with header_col2:
    page = st.selectbox(
        "Navigation", 
        ["Stress Detection Portal", "Analytics Dashboard"], 
        label_visibility="collapsed"
    )

with header_col1:
    if page == "Stress Detection Portal":
        st.markdown("### 🧠 Student Stress Detection System")
        st.markdown("###### Real-time Emotion & Stress Tracking")
    else:
        st.markdown("### 📊 Student Stress Analytics Dashboard")
        st.write("")

st.write("---")

if page == "Stress Detection Portal":
    # Context-aware environment detection
    is_streamlit_cloud = os.path.exists("/mount/src")
    is_hugging_face = "SPACE_ID" in os.environ
    is_cloud = is_streamlit_cloud or is_hugging_face
    
    mode = st.radio(
        "Choose Input Mode",
        ["🎥 Real-time Webcam", "📁 Upload Image / Photo"],
        horizontal=True,
    )
    
    # Split layout 1:1 to keep columns balanced at 50:50
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if mode == "🎥 Real-time Webcam":
            st.markdown("#### 🎥 Webcam Feed")
            # Sub-columns to shrink the webcam window by 25% (taking 3/4 of the 50% column width)
            sub_col1, sub_col2 = st.columns([3, 1])
            with sub_col1:
                if is_cloud:
                    # Ensure camera index state exists
                    if "camera_index" not in st.session_state:
                        st.session_state.camera_index = 0
                    
                    camera_key = f"cloud_cam_{st.session_state.camera_index}"
                    
                    # Capture manual snapshot (fully visible live feed, hides when image is captured)
                    img = st.camera_input("Capture", key=camera_key, label_visibility="collapsed")
                    
                    if img:
                        # Once photo is taken, inject CSS to hide the raw camera widget completely
                        hide_camera_css = """
                        <style>
                        [data-testid="stCameraInput"] {
                            display: none !important;
                        }
                        </style>
                        """
                        st.markdown(hide_camera_css, unsafe_allow_html=True)
                        
                        # Process the frame
                        processed_frame, stress_val, emotion = process_cloud_frame(img.getvalue())
                        if processed_frame is not None:
                            st.image(processed_frame, use_column_width=True)
                        st.session_state.cloud_stress["counter"] = stress_val
                        
                        # Custom Retake button to go back to live camera feed
                        if st.button("📸 Retake Photo"):
                            st.session_state.camera_index += 1
                            st.rerun()
                    else:
                        st.info("💡 Position your face in the frame and click **Take Photo**.")
                else:
                    # Localhost: direct STUN P2P (fastest, no relay overhead)
                    frontend_config = server_config = {
                        "iceServers": [
                            {"urls": ["stun:stun.l.google.com:19302"]},
                            {"urls": ["stun:stun1.l.google.com:19302"]},
                            {"urls": ["stun:stun2.l.google.com:19302"]},
                        ]
                    }
                    
                    webrtc_streamer(
                        key="stress-detector",
                        video_frame_callback=video_frame_callback,
                        frontend_rtc_configuration=frontend_config,
                        server_rtc_configuration=server_config,
                        media_stream_constraints={"video": True, "audio": False},
                    )
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
        st.markdown("#### 📊 Status & Telemetry")
        
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
                    format="YYYY-MM-DD HH:mm:ss",
                    alignment="center"
                ),
                "Stress Level": st.column_config.NumberColumn(
                    "Stress Level",
                    format="%d%%",
                    alignment="center"
                )
            }
        )
    else:
        st.info("No logs saved yet. Please perform stress detection first.")
