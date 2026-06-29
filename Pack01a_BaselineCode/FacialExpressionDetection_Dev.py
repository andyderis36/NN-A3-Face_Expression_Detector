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
