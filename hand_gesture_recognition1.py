import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


IMG_SIZE = 64
DATA_PATH = r"C:\Users\parsuramulu\OneDrive\Desktop\Task 4\leapGestRecog"
GESTURE_CLASSES = {
    "05_thumb": 0,
    "06_index": 1,
    "07_ok": 2,
    "08_palm_moved": 3,
    "09_c": 4,
    "10_down": 5
}
label_to_name = {v: k for k, v in GESTURE_CLASSES.items()}

print("[INFO] Loading and preprocessing images...")

data, labels = [], []

for user_id in os.listdir(DATA_PATH):
    user_path = os.path.join(DATA_PATH, user_id)
    if os.path.isdir(user_path) and user_id == "04":
        for gesture_folder, label in GESTURE_CLASSES.items():
            folder_path = os.path.join(user_path, gesture_folder)
            if not os.path.exists(folder_path):
                continue
            for img_name in os.listdir(folder_path)[:100]:  # 
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    data.append(img)
                    labels.append(label)

data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
labels = to_categorical(np.array(labels), num_classes=len(GESTURE_CLASSES))

plt.figure(figsize=(12, 4))
for i in range(6):
    idx = np.where(np.argmax(labels, axis=1) == i)[0][0]
    plt.subplot(1, 6, i + 1)
    plt.imshow(data[idx].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(label_to_name[i])
    plt.axis('off')
plt.suptitle("Sample Images from Each Gesture Class")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(GESTURE_CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training model...")
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


loss, accuracy = model.evaluate(X_test, y_test)
print(f"[INFO] Test Accuracy: {accuracy * 100:.2f}%")


y_pred = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred_labels = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred_labels, target_names=list(label_to_name.values())))

cm = confusion_matrix(y_true, y_pred_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=label_to_name.values(), yticklabels=label_to_name.values())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


model.save("gesture_recognition_model.h5")
print("[INFO] Model saved as gesture_recognition_model.h5")
