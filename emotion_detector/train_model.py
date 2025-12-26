import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import os

# Configuration
width, height = 48, 48
num_features = 64
num_labels = 7
batch_size = 64
epochs = 50
data_path = 'fer2013.csv'  # Expected to be in the same directory or provide path

def load_data(path):
    if not os.path.exists(path):
        print(f"Dataset not found at {path}. Please download 'fer2013.csv' from Kaggle.")
        return None, None, None, None

    print(f"Loading data from {path}...")
    data = pd.read_csv(path)

    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = [int(pixel) for pixel in pixel_sequence.split(' ')]
        face = np.asarray(face).reshape(width, height)
        face = cv2.resize(face.astype('uint8'), (width, height))
        faces.append(face.astype('float32'))

    faces = np.asarray(faces)
    faces = np.expand_dims(faces, -1)

    # Normalize data to be between 0 and 1
    faces = faces / 255.0

    emotions = to_categorical(data['emotion'], num_classes=num_labels)

    return faces, emotions

def create_model():
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_initializer='he_normal'))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2*2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))

    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    import cv2 # Import here to avoid dependency if not needed for loading

    # Try to load data
    X, y = load_data(data_path)

    if X is not None:
        # Split data (simple split)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

        model = create_model()
        print("Starting training...")
        model.fit(np.array(X_train), np.array(y_train),
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(np.array(X_test), np.array(y_test)),
                  shuffle=True)

        # Save model
        model_json = model.to_json()
        with open("emotion_detector/emotion_model.json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights("emotion_detector/emotion_model.h5")
        print("Model saved to emotion_detector/emotion_model.json and emotion_detector/emotion_model.h5")
    else:
        print("Skipping training. Please provide 'fer2013.csv' to train the model.")
