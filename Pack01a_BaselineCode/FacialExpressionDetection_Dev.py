#STINK 3014 - Neural Networks
#Chap 06 - Deep Learning (Stress Detection Model - to generate baseline pre-trained model stress_detector_cnn.h5)

import os
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
output_dir = os.path.join(root_dir, "output")
os.makedirs(output_dir, exist_ok=True)

# data source
csv_path = os.path.join(current_dir, "fer2013.csv")
data = pd.read_csv(csv_path)
images = []
labels = []

# Select a record by index
record_index = 10  # change this to view any row
row = data.iloc[record_index]

# Get emotion label
emotion_label = int(row['emotion'])

# Convert pixels string to numpy array
pixels = np.array(row['pixels'].split(), dtype=np.uint8)
img = pixels.reshape(48,48)  # FER2013 images are 48x48

# Display using matplotlib
plt.imshow(img, cmap='inferno')
plt.title(f'FER2013 Record {record_index} - Emotion {emotion_label}')
plt.axis('off')
plt.show()


# FER2013 mapping: 0=Angry,1=Disgust,2=Fear,3=Happy,4=Sad,5=Surprise,6=Neutral
# We use all 7 classes from the FER2013 dataset
for index, row in data.iterrows():
    emotion = int(row['emotion'])
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    img = pixels.reshape(48,48)
    images.append(img)
    labels.append(emotion)
X = np.array(images).reshape(-1,48,48,1)/255.0
y = to_categorical(labels, num_classes=7)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

model = Sequential([
    Input(shape=(48, 48, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'), # Extra Conv layer for feature extraction depth
    MaxPooling2D((2, 2)),                   # Extra Pooling layer
    Flatten(),
    Dense(256, activation='relu'),          # Increased capacity of Dense layer
    Dropout(0.5),                           # Increased Dropout for regularization
    Dense(7, activation='softmax')          # 7 outputs for full FER2013 classification
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train CNN
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.2)

# Plot Training History
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plot_save_path = os.path.join(output_dir, "training_history.png")
plt.savefig(plot_save_path)
print(f"Training history plot saved as {plot_save_path}")
plt.show()

# Save trained model
model_save_path = os.path.join(output_dir, "emotion_stress_cnn.h5")
model.save(model_save_path)
print(f"Model saved as {model_save_path}")

# --- EVALUATION BLOCK (Data Scientist additions) ---
from sklearn.metrics import confusion_matrix

print("\n--- Model Evaluation on Test Set ---")
# 1. Predict on X_test and extract argmax
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 2. Binary mapping: Stress (0 = Angry, 2 = Fear) vs Non-Stress (others)
# FER2013 mapping: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
y_true_binary = np.isin(y_true, [0, 2]).astype(int)
y_pred_binary = np.isin(y_pred, [0, 2]).astype(int)

# Calculate confusion matrix components
tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()

# Print CM table in terminal
print("\nConfusion Matrix (Stress vs Non-Stress):")
print(f"{'':<18} | {'Predicted Non-Stress':<20} | {'Predicted Stress':<20}")
print("-" * 66)
print(f"{'Actual Non-Stress':<18} | {f'TN: {tn}':<20} | {f'FP: {fp}':<20}")
print(f"{'Actual Stress':<18} | {f'FN: {fn}':<20} | {f'TP: {tp}':<20}")

# 3. Calculate metrics
total = tn + fp + fn + tp
accuracy = (tp + tn) / total if total > 0 else 0
stress_detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

print("\nEvaluation Metrics:")
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Stress Detection Rate (Recall): {stress_detection_rate * 100:.2f}%")
print(f"False Alarm Rate: {false_alarm_rate * 100:.2f}%")

# 4. Save CM Visualization Plot
plt.figure(figsize=(6, 5))
cm_data = [[tn, fp], [fn, tp]]
im = plt.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Stress vs Non-Stress)')
plt.colorbar(im)
classes = ['Non-Stress', 'Stress']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add text labels inside the cells
thresh = (tn + fp + fn + tp) / 4
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm_data[i][j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm_data[i][j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()

cm_save_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(cm_save_path)
print(f"\nConfusion matrix visualization saved as {cm_save_path}")
plt.show()

