#STINK 3014 Neural Networks
#Chap 06 Deep Learning - Stress Detection App. + using generator model (stress_detector_cnn.h5)

#import libraries
import cv2
import tensorflow as tf
import numpy as np

#load the CNN model (from previous CNN trained model)
model = tf.keras.models.load_model("stress_detector_cnn.h5")
labels_dict = {0: "Calm", 1: "Stressed"}

#cv2.VideoCapture(0) - value 0 for the default-built in webcam
cap = cv2.VideoCapture(0)

#haarcascades - pretrained HAAR Cascade Object detection model (in this case to detect the frontal face region)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#display stress / calm results
def draw_emotion_bars(frame, x, y, predictions):

    #Draw a mini bar chart next to the detected face.
    bar_width = 100         # width of probability chart
    bar_height = 20         # height of each bar
    spacing = 35             # gap between bars
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
        color = (0, 255, 0) if i == 0 else (0, 0, 255)

        cv2.rectangle(frame,
                      (start_x, start_y + i * (bar_height + spacing)),
                      (start_x + bar_length, start_y + i * (bar_height + spacing) + bar_height),
                          color, -1)

        # Label + probability text
        cv2.putText(frame,
                    f"{label}: {prob_percent}%",
                    (start_x, start_y + i * (bar_height + spacing) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255,255,255), 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi / 255.0
        roi = roi.reshape(1,48,48,1)

        predictions = model.predict(roi)[0]
        class_id = np.argmax(predictions)
        conf = predictions[class_id]

        # Draw bounding box
        color = (0,255,0) if class_id == 0 else (0,0,255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Draw label
        text = f"{labels_dict[class_id]} ({conf*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Draw emotion probability bars
        draw_emotion_bars(frame, x, y, predictions)

    cv2.imshow("Stress Detection with Probabilities", frame)

# Press q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
