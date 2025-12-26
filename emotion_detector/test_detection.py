from detect_emotion import detect_emotion_image
import os

if __name__ == "__main__":
    if os.path.exists('emotion_detector/test.jpg'):
        print("Running detection on test.jpg...")
        detect_emotion_image('emotion_detector/test.jpg', output_path='emotion_detector/test_result.jpg')
    else:
        print("test.jpg not found")
