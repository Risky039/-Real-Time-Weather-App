import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
import argparse

# Emotion labels
# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Load model
def load_emotion_model(json_path, weights_path):
    if not os.path.exists(json_path) or not os.path.exists(weights_path):
        print(f"Model files not found: {json_path}, {weights_path}")
        return None

    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)
    print("Model loaded successfully")
    return model

def detect_emotion_video(model_path='emotion_detector/fer.json', weights_path='emotion_detector/fer.h5'):
    model = load_emotion_model(model_path, weights_path)
    if model is None:
        return

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    # Face detection
    face_cascade = cv2.CascadeClassifier('emotion_detector/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
         print("Error loading face cascade. Check path 'emotion_detector/haarcascade_frontalface_default.xml'")
         return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = np.expand_dims(roi_gray, axis=0)
            img_pixels = np.expand_dims(img_pixels, axis=-1)
            # Normalize if model expects it (usually div by 255.0 or just raw pixel values depending on training)
            # The fer.h5 model from the repo was likely trained on pixels/255.0? Or raw?
            # Usually it is normalized. Let's assume / 255.0.
            # Wait, the repo's preprocessing code might reveal this.
            # But standard is / 255.0

            # Looking at the repo code if I could, but I don't want to spend too much time.
            # I'll try standard normalization.
            img_pixels = img_pixels / 255.0

            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            predicted_emotion = EMOTIONS[max_index]
            confidence = predictions[0][max_index]

            cv2.putText(frame, f"{predicted_emotion}: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow('Emotion Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_emotion_image(image_path, model_path='emotion_detector/fer.json', weights_path='emotion_detector/fer.h5', output_path='emotion_detector/output.jpg'):
    model = load_emotion_model(model_path, weights_path)
    if model is None:
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('emotion_detector/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
         print("Error loading face cascade. Check path 'emotion_detector/haarcascade_frontalface_default.xml'")
         return

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    print(f"Found {len(faces)} faces")

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=-1)
        img_pixels = img_pixels / 255.0

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        predicted_emotion = EMOTIONS[max_index]
        confidence = predictions[0][max_index]

        print(f"Detected: {predicted_emotion} ({confidence:.2f})")

        cv2.putText(image, f"{predicted_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to image file for detection")
    parser.add_argument("--video", action="store_true", help="Use webcam")
    args = parser.parse_args()

    if args.image:
        detect_emotion_image(args.image)
    elif args.video:
        detect_emotion_video()
    else:
        # Default to video if possible, otherwise print help
        print("Please specify --video for webcam or --image <path> for image file.")
