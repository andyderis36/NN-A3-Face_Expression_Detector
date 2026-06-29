---
title: Face Expression Detector
emoji: 🧠
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: 1.30.0
python_version: 3.12
app_file: app.py
pinned: false
---

# Student Stress & Emotion Detection System (STINK3014 - Assignment 3)

Comprehensive implementation for UUM Neural Networks course (STINK3014) on **"Facial Images-Based Stress and Emotion Detection Using Deep Learning Technique"**.

This repository contains a full pipeline: training a 7-class CNN model on the FER2013 dataset, deploying a local real-time OpenCV-based webcam application, and hosting a production-grade interactive WebRTC Streamlit Dashboard suitable for cloud deployment.

---

## 📂 Repository Structure

```text
A3/
├── Pack01a_BaselineCode/
│   ├── FacialExpressionDetection_Dev.py  # Model training, validation, and history plotting
│   └── fer2013.csv                       # FER2013 dataset file (Download: https://drive.google.com/file/d/1rYHyg1HR84RycynB0BQ6K2RNE6dwDX_4/view?usp=sharing)
├── Pack01b_DeploymentCode/
│   ├── FacialExpressionDetection_App.py  # Local OpenCV-based webcam application
│   └── haarcascade_frontalface_default.xml # Haar Cascade face detector
├── Pack02_SUEQ tool/
│   ├── Short_UEQ_Data_Analysis_Tool.xlsx # S-UEQ data processing tool
│   └── UEQS_Items.pdf                    # Bipolar 7-point scale questionnaires
├── docs/
│   ├── Assignment03_Part1_Report.md      # Part I Report (Technical & Evaluation Answers)
│   └── Assignment03_Part2_Report.md      # Part II Report (Student Mental Health Architecture)
├── output/                               # Generated runtime outputs (auto-created)
│   ├── emotion_stress_cnn.h5             # Saved 7-class CNN model
│   ├── training_history.png              # Accuracy and Loss curves
│   └── stress_log.csv                    # Logs of high stress events
├── app.py                                # Streamlit WebRTC Dashboard (Main Cloud Entry)
├── requirements.txt                      # Pip dependencies for cloud deployment
└── README.md                             # Repository documentation
```

---

## 📈 Model Evaluation & Training Results

Below are the accuracy and loss curves generated after training the 7-class CNN model for 15 epochs on the FER2013 dataset:

![Model Training History](output/training_history.png)

---

## ⚙️ Environment Setup & Installation

All steps assume execution from the root of the repository.

### 1) Activate Virtual Environment
Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```
Windows (CMD):
```cmd
.\.venv\Scripts\activate.bat
```

### 2) Install Dependencies
For local and cloud deployment, run:
```powershell
pip install -r requirements.txt
```

---

## 🚀 Execution Guide

### Step 1: Train the CNN Model (Baseline & History)
Run the training script to load all 7 classes of the FER2013 dataset, build/compile the deepened CNN, and export accuracy curves:
```powershell
.\.venv\Scripts\python.exe "Pack01a_BaselineCode\FacialExpressionDetection_Dev.py"
```
*Outputs generated in `output/`: `emotion_stress_cnn.h5` and `training_history.png`.*

### Step 2: Run Local OpenCV Webcam Application
Run the local camera detection utility that uses a temporal counter, visual progress bars, audio beep alerts (Windows native), and CSV event logging:
```powershell
.\.venv\Scripts\python.exe "Pack01b_DeploymentCode\FacialExpressionDetection_App.py"
```
*Press `q` on the camera window to exit.*

### Step 3: Run Interactive Streamlit Dashboard (Local / Web)
Run the web-based WebRTC application that operates asynchronously and compiles logs:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
*Access via your browser at: `http://localhost:8501`*

---

## ☁️ Deployment to Streamlit Community Cloud

This project is fully compatible with Streamlit Community Cloud:
1. **GitHub Push**: Commit and push the entire repository (including `app.py`, `requirements.txt`, `output/emotion_stress_cnn.h5`, and the `Pack01b_DeploymentCode/` folder).
2. **Streamlit Cloud Connection**:
   * Visit [share.streamlit.io](https://share.streamlit.io/).
   * Select your repository and branch (`main`).
   * Set **Main file path** to `app.py`.
   * Click **Deploy**.
3. The platform will read `requirements.txt`, install dependencies (running tensorflow-cpu and headless-opencv), and serve the app live with WebRTC webcam support!

---

## ⚠️ Git Configuration & Large Files (.gitignore)

A pre-configured `.gitignore` file is provided in this repository. It is **critical** to verify this before pushing:
* **Large Dataset (`fer2013.csv`)**: The FER2013 dataset is approximately **300MB**. GitHub has a strict file size limit of **100MB** per file. Trying to push this file will cause your git push to fail. The `.gitignore` is set to ignore `**/fer2013.csv`. You can download it from [Google Drive](https://drive.google.com/file/d/1rYHyg1HR84RycynB0BQ6K2RNE6dwDX_4/view?usp=sharing) and place it inside the `Pack01a_BaselineCode/` directory to run training locally.
* **Virtual Environment (`.venv/`)**: Prevents uploading local packages. Streamlit Cloud will install these dynamically.
* **Local Logs (`output/stress_log.csv`)**: Excluded from version control so that cloud logging runs independently.

---

## 🛠️ Code Customizations & Features

* **7-Class Classifier**: Trained on standard FER2013 classes (0: Angry, 1: Disgust, 2: Fear, 3: Happy, 4: Sad, 5: Surprise, 6: Neutral) to enable accurate non-stressed states (like smiling or being neutral) and prevent false stress alarms.
* **Temporal Stress Accumulation**: Level increases with consecutive Angry/Fear detections and decays when Calm (Happy/Neutral/Surprise) is detected.
* **HUD Overlay**: Clean, anti-aliased text overlay (`cv2.LINE_AA`) displaying stress meters next to the face and warning signs.
* **Time-Window Filtering**: Dashboard analytics chart allows filtering logged data to view the last 5, 10, 15, 30 minutes, or 1 hour dynamically.
* **STUN Server ICE Configuration**: Integrated Google's public STUN server (`stun:stun.l.google.com:19302`) in `app.py` WebRTC configuration to ensure connection traversal works seamlessly across firewalls and NATs on Streamlit Cloud.
