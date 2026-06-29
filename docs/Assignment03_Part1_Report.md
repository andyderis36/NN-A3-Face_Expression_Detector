# STINK3014 (Neural Networks) - Assignment 3 Report
## Part I: Baseline Model Development and Deployment

**Student Name:** [Your Name]  
**Course Code:** STINK3014 (Neural Networks)  
**Instructor:** Associate Prof. Dr. Azizi Ab Aziz  
**Project Title:** Facial Images-Based Stress and Emotion Detection Using Deep Learning Technique

---

## 1. Introduction and Objectives
The objective of this assignment is to develop a real-time facial stress and emotion detection system using a Convolutional Neural Network (CNN). The system processes facial images via a webcam, detects the face area, classifies the emotion, tracks stress levels over consecutive frames, provides visual and audio feedback, and logs occurrences to a CSV file.

All technical requirements of **Part I (A. Training the Model)** and **B. Deployment & Real-Time Detection** have been successfully implemented and verified in the repository.

---

## 2. Technical Modifications & Implementations

### 2.1 Model Architecture Tuning (`FacialExpressionDetection_Dev.py`)
To improve classification performance beyond the baseline (~63%), the CNN architecture was optimized and refactored:
1. **Input Layer Warning Fix**: Replaced the legacy `input_shape` parameter in the first Conv2D layer with an explicit `tf.keras.layers.Input(shape=(48, 48, 1))` layer.
2. **Deepening the Network**: Added a third convolution block (`Conv2D(128) + MaxPooling2D`) to capture higher-level facial features and spatial representations.
3. **Capacity & Regularization**: Increased the fully connected dense layer capacity to `256` units and raised `Dropout` to `0.5` to reduce overfitting during the 15 training epochs.
4. **SoftMax Classifier**: Standardized on a 7-output SoftMax classifier representing the full FER2013 categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
5. **Training History Visualization**: Configured the script to automatically plot Training vs. Validation Accuracy and Loss, saving the result as `training_history.png` in the `output/` directory.

### 2.2 Webcam Deployment (`FacialExpressionDetection_App.py`)
1. **Consecutive Frame Tracking**: Implemented a `stress_counter` (capped at 100%) that increments when stress emotions are detected and decays gradually when not detected.
2. **Stress Level Meter**: Created a dynamic visual progress bar drawn directly on the OpenCV window showing the accumulated stress percentage.
3. **Non-blocking Audio Alerts**: Configured `winsound.Beep` to run in a separate asynchronous thread (preventing camera lag) with a 2-second cooldown to avoid sound spamming.
4. **CSV Logging**: Programmed automatic logging of timestamps and stress levels to `output/stress_log.csv` only when the stress level exceeds a threshold of 30.

### 2.3 Cloud-Native Streamlit Dashboard (`app.py`)
To enable public web hosting and demonstrate deployment-readiness, a Streamlit dashboard was created:
1. **WebRTC Integration**: Used `streamlit-webrtc` with a thread-safe frame processing callback to fetch, process, and return annotated video frames asynchronously on the server.
2. **HUD Bounding Box Overlay**: Renders real-time classification, smooth progress bar meters, and alert markers directly on the browser feed.
3. **Analytics Panel**: Includes an interactive Plotly chronological stress graph with customized selection windows (Last 5, 10, 15, 30 minutes, or 1 hour) and a centered data grid log.

---

## 3. Answers to Report Questions (Part I)

### A. Conceptual Questions

#### 1. What are the main stress-related emotions in this project, and why are they selected?
* **Emotions**: The main emotions selected are **Angry (Marah)** and **Fear (Takut)**.
* **Rationale**: These emotions are physiologically and psychologically linked to the human stress response ("fight or flight"). Anger activates the sympathetic nervous system, causing physical strain. Fear triggers anxiety and high alertness. Both emotions are reliable indicators of acute psychological stress, making them excellent targets for visual stress detection.

#### 2. Explain the role of CNN layers in feature extraction for facial emotion detection.
* **Conv2D Layers**: Apply learnable kernels to detect local patterns. Shallow layers detect edges and textures, while deeper layers detect complex shapes like the corners of eyes, eyebrows, and mouth curvature.
* **Activation (ReLU)**: Introduces non-linearity to learn complex non-linear boundaries.
* **MaxPooling2D**: Downsamples spatial dimensions, reducing computational load and providing translation invariance (allowing the network to detect features regardless of their exact position).
* **Flatten & Dense**: Combine extracted spatial features to perform final classification.

#### 3. How does the system reduce false alarms for stress/emotion detection?
* The system uses a **consecutive frame counter**. Single-frame misclassifications due to facial movement or lighting changes are filtered out. Only when stress-related emotions are detected consistently across multiple consecutive frames (e.g., 30 frames) does the system register "High Stress" and trigger alerts.

#### 4. Discuss ethical considerations when deploying real-time stress detection.
* **Privacy**: Processing real-time webcam feeds requires explicit user consent. Raw camera images should be processed entirely locally (edge processing) and never stored or transmitted.
* **Data Anonymization**: The output CSV log must not contain personally identifiable information (PII) like names or raw images—only timestamps and numerical levels.
* **Bias & Fairness**: The facial expression model must be trained on diverse datasets to prevent bias across different demographics, ages, and genders.

#### 5. How can audio alerts and CSV logging enhance user awareness and research analysis?
* **Audio Alerts**: Provide immediate real-time feedback, prompting the user to take a break or perform breathing exercises when high stress levels are sustained.
* **CSV Logging**: Creates a structured timeline for researcher/user analysis. Users can review logs to discover patterns (e.g., higher stress during specific times of day or tasks) to manage their well-being.

---

### B. Technical Questions

#### 1. Explain the preprocessing steps for FER2013 images.
1. **Parsing**: Convert the pixel string (2304 values) into a 2D numpy array of shape `(48, 48)`.
2. **Grayscale**: Ensure the image remains single-channel (grayscale).
3. **Normalization**: Divide pixel values by `255.0` to rescale them from `[0, 255]` to `[0.0, 1.0]`. This speeds up gradient descent convergence.
4. **Target Encoding**: Convert numerical labels into categorical dummy variables using `to_categorical`.

#### 2. How is the stress level meter calculated from consecutive frame detections?
* When a face is detected:
  * If the predicted class is a stress-related emotion (Angry/Fear), the `stress_counter` increments by 1.
  * If the predicted class is NOT a stress-related emotion (or no face is detected), the `stress_counter` decrements by 1 (down to a minimum of 0).
* The counter is capped at 100 to represent a percentage meter.

#### 3. Why is SoftMax activation used in the output layer?
* SoftMax converts raw logits from the final Dense layer into a probability distribution over the classes. The outputs are squashed between `0` and `1`, and their sum equals `1.0`. This represents the model's confidence for each class.

#### 4. Describe how you would evaluate the model for stress detection accuracy.
* Evaluation metrics include:
  * **Classification Accuracy**: Percentage of correct predictions on the test split.
  * **Confusion Matrix**: Visualizing True Positives (correctly identified stress), False Positives (calm identified as stress), and False Negatives.
  * **Stress Detection Rate (Recall)**: Ability to capture actual stress.
  * **False Alarm Rate**: Rate at which non-stress is classified as stress.

---

### C. Evaluation / Experiment Questions

#### 1. Plot training & validation accuracy for your CNN model.
The training history plot (displaying accuracy and loss curves for both training and validation splits) is generated automatically upon running the training script:

![CNN Model Training History](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/NN/A3/output/training_history.png)

*The plot is also saved locally for submission at [output/training_history.png](file:///C:/Users/andyd/Documents/UUM/UUM%20OL/A252/NN/A3/output/training_history.png).*

#### 2. Provide a confusion matrix for stress vs non-stress detection.
* Can be generated on the test split using `sklearn.metrics.confusion_matrix`.

#### 3. Analyse stress detection rate vs false alarm rate.
* A low detection threshold captures stress quickly (high recall) but increases false alarms. A higher threshold reduces false alarms but might delay or miss stress detection. Tuning the consecutive frame threshold (e.g. 30 frames) balances this trade-off.

#### 4. Discuss limitations of using only facial images for stress detection.
* **Force-Classifying**: In a binary system trained only on Angry vs. Fear, any face (including neutral or happy) is forced into one of these classes. (Note: This limitation was resolved by migrating to a 7-class emotion model, where neutral or smiling faces are correctly classified as "Neutral" or "Happy" and do not falsely trigger stress accumulation.)
* **Environmental Factors**: Poor lighting, head angles, or glasses can obstruct face landmarks, causing false detections.
* **Alternative Expression Patterns**: People express emotions differently; a neutral face of one person might look "angry" or "fearful" to the model due to physiological facial structure.

#### 5. Suggest real-world applications of your system and potential improvements.
* **Applications**: Driver fatigue/stress monitoring, online learning attention tracking, workspace wellness tools.
* **Improvements**:
  1. Add a "Neutral" or "Calm" class during training so the model can predict "No Stress" explicitly.
  2. Integrate multi-modal signals (e.g., heart rate from webcam photoplethysmography or voice tone).
