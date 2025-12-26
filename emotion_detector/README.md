# Real-Time Emotion Detector

This project implements a Real-Time Emotion Detector using Python, TensorFlow/Keras, and OpenCV. It captures video from a webcam (or processes an image), detects faces, and classifies the expression into one of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.

## Setup

1.  **Install Dependencies:**

    ```bash
    pip install tensorflow opencv-python matplotlib numpy pandas scikit-learn
    ```

2.  **Dataset (Optional for Training):**

    To train the model from scratch, you need the FER-2013 dataset.
    *   Download `fer2013.csv` from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
    *   Place `fer2013.csv` in the `emotion_detector/` directory.

## Usage

### 1. Real-time Detection (Webcam)

To run the emotion detector using your webcam:

```bash
python3 emotion_detector/detect_emotion.py --video
```

### 2. Image Detection

To detect emotions in a static image:

```bash
python3 emotion_detector/detect_emotion.py --image <path_to_image>
```

Example:

```bash
python3 emotion_detector/detect_emotion.py --image emotion_detector/test.jpg
```

### 3. Training the Model (Optional)

If you have the `fer2013.csv` dataset, you can train the model:

```bash
python3 emotion_detector/train_model.py
```

This will save the trained model to `emotion_detector/emotion_model.json` and `emotion_detector/emotion_model.h5`.

## Files

*   `emotion_detector/detect_emotion.py`: Main script for emotion detection (video/image).
*   `emotion_detector/train_model.py`: Script to train the CNN model.
*   `emotion_detector/fer.json` & `emotion_detector/fer.h5`: Pre-trained model architecture and weights.
*   `emotion_detector/haarcascade_frontalface_default.xml`: OpenCV Haar Cascade for face detection.
