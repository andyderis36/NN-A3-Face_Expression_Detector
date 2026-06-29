#STINK 3014 Neural Networks
#Chap 06 Deep Learning - Stress Detection App. + using generator model (stress_detector_cnn.h5)

#import libraries
import cv2
import tensorflow as tf
import numpy as np
import winsound
import csv
import datetime
import time
import threading
import os

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# Initialize CSV log file with header if it doesn't exist
log_file = os.path.join(output_dir, "stress_log.csv")
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Stress Level"])

#load the CNN model (from output directory)
model_path = os.path.join(output_dir, "emotion_stress_cnn.h5")
model = tf.keras.models.load_model(model_path)
labels_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

# Stress logic variables
stress_counter = 0
STRESS_ALERT_THRESHOLD = 30  # High stress threshold (approx 30 consecutive frames)
last_alert_time = 0

#cv2.VideoCapture(0) - value 0 for the default-built in webcam
cap = cv2.VideoCapture(0)

#haarcascades - pretrained HAAR Cascade Object detection model (in this case to detect the frontal face region)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#display stress / calm results
def draw_emotion_bars(frame, x, y, w, predictions):

    #Draw a mini bar chart next to the detected face.
    bar_width = 100         # width of probability chart
    bar_height = 12         # height of each bar
    spacing = 15             # gap between bars
    start_x = x + w + 15    # position of chart (right of face)
    start_y = y

    for i, prob in enumerate(predictions):
        label = labels_dict[i]
        prob_percent = int(prob * 100)

        # Bar background (grey)
        cv2.rectangle(frame,
                      (start_x, start_y + i * (bar_height + spacing)),
                      (start_x + bar_width, start_y + i * (bar_height + spacing) + bar_height),
                      (180, 180, 180), -1)

        # Bar foreground — proportional length
        bar_length = int(bar_width * prob)
        # Red for stress-related (Angry=0, Fear=2), Green for others
        color = (0, 0, 255) if i in [0, 2] else (0, 255, 0)

        cv2.rectangle(frame,
                      (start_x, start_y + i * (bar_height + spacing)),
                      (start_x + bar_length, start_y + i * (bar_height + spacing) + bar_height),
                          color, -1)

        # Label + probability text
        cv2.putText(frame,
                    f"{label}: {prob_percent}%",
                    (start_x, start_y + i * (bar_height + spacing) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255,255,255), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # If no face is detected, we can gradually decrease stress counter
    if len(faces) == 0:
        stress_counter = max(0, stress_counter - 1)
        
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi / 255.0
        roi = roi.reshape(1,48,48,1)

        predictions = model.predict(roi)[0]
        class_id = np.argmax(predictions)
        conf = predictions[class_id]

        # Stress accumulation logic: Angry (0) and Fear (2) are stress emotions in FER2013
        if class_id in [0, 2]: 
            stress_counter += 1
        else:
            stress_counter = max(0, stress_counter - 1)
            
        stress_counter = min(100, stress_counter) # Cap at 100%

        # Draw bounding box (Red for Stressed, Green for Calm/Others)
        color = (0,0,255) if class_id in [0, 2] else (0,255,0)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw label
        text = f"{labels_dict[class_id]} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw Stress Level Meter
        meter_x = x
        meter_y = y - 30
        cv2.rectangle(frame, (meter_x, meter_y - 15), (meter_x + 100, meter_y), (180,180,180), -1)
        cv2.rectangle(frame, (meter_x, meter_y - 15), (meter_x + stress_counter, meter_y), (0,0,255), -1)
        cv2.putText(frame, f"Stress: {stress_counter}%", (meter_x, meter_y - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255) if stress_counter >= STRESS_ALERT_THRESHOLD else (255,255,255), 2)

        # Audio Alert and CSV Logging
        if stress_counter >= STRESS_ALERT_THRESHOLD:
            cv2.putText(frame, "HIGH STRESS ALERT!", (x, y - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            current_time = time.time()
            if current_time - last_alert_time > 2.0: # Cooldown of 2 seconds
                threading.Thread(target=winsound.Beep, args=(1000, 300)).start()
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(log_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([timestamp, stress_counter])
                last_alert_time = current_time

        # Draw emotion probability bars
        draw_emotion_bars(frame, x, y, w, predictions)

    cv2.imshow("Stress Detection with Probabilities", frame)

# Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
