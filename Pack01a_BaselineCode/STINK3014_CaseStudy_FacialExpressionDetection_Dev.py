#STINK 3014 - Neural Networks
#Chap 06 - Deep Learning (Stress Detection Model - to generate baseline pre-trained model stress_detector_cnn.h5)

import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
import pandas as pd
import matplotlib.pyplot as plt

# data source
data = pd.read_csv("fer2013.csv")
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
stress_classes = [0,2,4]  # Angry, Fear, Sad -> stress
calm_classes   = [3,5,6]  # Happy, Surprise, Neutral -> calm

for index, row in data.iterrows():
    emotion = int(row['emotion'])
    pixels = np.array(row['pixels'].split(), dtype=np.uint8)
    img = pixels.reshape(48,48)

    if emotion in stress_classes:
        label = 1  # Stress
    else:
        label = 0  # Calm
    images.append(img)
    labels.append(label)
X = np.array(images).reshape(-1,48,48,1)/255.0
y = to_categorical(labels, num_classes=2)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32,(3,3),activation='relu', input_shape=(48,48,1)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train CNN
model.fit(X_train, y_train, epochs=1, batch_size=32, validation_split=0.2)

# Save trained model
model.save("stress_detector_cnn.h5")
print("Model saved as stress_detector_cnn.h5")
