import os
import cv2
import math
import logging
import geocoder
import numpy as np
import pandas as pd
import datetime
import base64

from typing import List, Dict, Any
from geopy.geocoders import Nominatim

# Tensorflow and ML libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AccidentDetectionSystem:
    def __init__(self, camera_ips: List[str], model_path: str = 'accident_detection_model.h5'):
        self.camera_ips = camera_ips
        self.model_path = model_path
        self.base_model = None
        self.model = None
        self.load_or_create_model()

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info("Initializing Accident Detection System")
        logger.info(f"Configured Cameras: {self.camera_ips}")

    def load_or_create_model(self):
        """Load existing model or create and save a new one"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        else:
            logger.info("No existing model found. Creating new model.")
            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
            self.model = self.build_model()

    def extract_video_frames(self, video_path: str, output_path: str) -> None:
        """Extract frames from video"""
        logger.info(f"Extracting frames from {video_path}")
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(5)
        count = 0

        os.makedirs(output_path, exist_ok=True)
        while cap.isOpened():
            frame_id = cap.get(1)
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_id % math.floor(frame_rate) == 0:
                filename = os.path.join(output_path, f"{count}.jpg")
                cv2.imwrite(filename, frame)
                count += 1
        
        cap.release()
        logger.info(f"Extracted {count} frames")

    def prepare_data(self, data_path: str, image_size: tuple = (224, 224)) -> tuple:
        """Prepare image data for training"""
        logger.info("Preparing training data")
        data = pd.read_csv('mapping.csv')
        X, y = [], []

        for img_name in data.Image_ID:
            img_path = os.path.join(data_path, img_name)
            img = load_img(img_path, target_size=image_size)
            img_array = img_to_array(img)
            X.append(img_array)
            y.append(data[data.Image_ID == img_name].Class.values[0])

        X = np.array(X)
        y = to_categorical(y)
        
        return X, y

    def build_model(self, input_shape=(224, 224, 3)) -> Sequential:
        """Build neural network model"""
        logger.info("Building machine learning model")
        
        model = Sequential([
            InputLayer((7*7*512,)),
            Dense(units=1024, activation='sigmoid'),
            Dense(2, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train_model(self, X, y, test_size=0.3, force_retrain=False):
        """Train the model and save it"""
        # If model exists and not forced to retrain, skip training
        if os.path.exists(self.model_path) and not force_retrain:
            logger.info("Existing model will be used. Set force_retrain=True to retrain.")
            return

        logger.info("Training machine learning model")
        X = preprocess_input(X)
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42)
        
        X_train_features = self.base_model.predict(X_train)
        X_valid_features = self.base_model.predict(X_valid)
        
        X_train_features = X_train_features.reshape(X_train_features.shape[0], 7*7*512)
        X_valid_features = X_valid_features.reshape(X_valid_features.shape[0], 7*7*512)
        
        X_train_features /= X_train_features.max()
        X_valid_features /= X_train_features.max()
        
        # Train model
        history = self.model.fit(
            X_train_features, y_train, 
            epochs=100, 
            validation_data=(X_valid_features, y_valid)
        )
        
        # Save the model
        save_model(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return history

    def get_geolocations(self) -> Dict[str, str]:
        """Fetch geo locations for camera IPs"""
        geolocator = Nominatim(user_agent="AccidentSystem")
        camera_locations = {}
        
        for ip in self.camera_ips:
            location = geocoder.ip(ip)
            if location.latlng:
                reverse_loc = geolocator.reverse(location.latlng)
                camera_locations[ip] = reverse_loc.address if reverse_loc else "Unknown Location"
                logger.info(f"Camera IP {ip}: {camera_locations[ip]}")
        
        return camera_locations

    def detect_accidents(self, video_path: str) -> None:
        """Detect accidents in video stream"""
        logger.info(f"Starting accident detection for {video_path}")
        cap = cv2.VideoCapture(video_path)
        i = 0
        flag = 0
        snapshot_counter = 0
        imgflag = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame for prediction
            resized_frame = cv2.resize(frame, (224, 224))
            preprocessed_frame = preprocess_input(np.expand_dims(resized_frame, axis=0))
            
            # Extract features
            frame_features = self.base_model.predict(preprocessed_frame)
            frame_features = frame_features.reshape(1, 7*7*512)
            frame_features /= frame_features.max()
            
            # Predict accident
            predictions = self.model.predict(frame_features)

            # Use model prediction for accident detection
            prediction_index = 0  # Since we're processing one frame at a time
            if predictions[prediction_index][0] < predictions[prediction_index][1]:
                percent = predictions[prediction_index][1] * 100
                predict = "No Accident"
            else:
                percent = predictions[prediction_index][0] * 100
                predict = f"Accident {percent:.2f}%"
                flag = 1

                # Capture snapshot if confidence > 60%
                if imgflag == 0 and percent > 60:
                    AccSnapshotDir = 'AccSnaps/'
                    os.makedirs(AccSnapshotDir, exist_ok=True)
                    snapshot_filename = f'accident_snapshot_{snapshot_counter}.jpg'
                    cv2.imwrite(os.path.join(AccSnapshotDir, snapshot_filename), frame)
                    snapshot_counter += 1
                    imgflag = 1

            # Display prediction on frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, predict, (50, 50), font, 1, (0, 255, 255), 3, cv2.LINE_4)
            
            cv2.imshow('Frame', frame)
            i += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    # Sample camera IPs (these would be real in production)
    camera_ips = ['192.168.1.100']
    
    # Initialize system
    accident_system = AccidentDetectionSystem(camera_ips)
    accident_system.setup_logging()
    
    # Extract training frames
    accident_system.extract_video_frames('Accidents.mp4', './traindata')
    
    # Prepare and train model
    X, y = accident_system.prepare_data('./traindata')
    accident_system.train_model(X, y)
    
    # Get camera locations
    camera_locations = accident_system.get_geolocations()
    
    # Detect accidents
    accident_system.detect_accidents('Accident-2.mp4')

if __name__ == "__main__":
    main()