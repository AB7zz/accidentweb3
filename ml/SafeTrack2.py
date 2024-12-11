import os
import cv2
import math
import logging
import geocoder
import numpy as np
import pandas as pd
import datetime
import base64
import requests
import random
import numpy as np

from typing import List, Dict, Any
from geopy.geocoders import Nominatim
from twilio.rest import Client
from dotenv import load_dotenv

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow and ML libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def set_seeds(seed=42):
    """
    Set seeds for reproducibility across different libraries
    
    :param seed: Seed value (default: 42)
    """
    try:
        # Ensure seed is converted to an integer
        seed = int(seed)
    except (TypeError, ValueError):
        # Fallback to a default seed if conversion fails
        seed = 42
    
    # Set seeds for different libraries
    try:
        # Python's built-in random
        random.seed(seed)
        
        # Numpy
        np.random.seed(seed)
        
        # TensorFlow
        tf.random.set_seed(seed)
        
        # Optional: For torch if you're using it
        # import torch
        # torch.manual_seed(seed)
        
        print(f"Random seeds set to {seed}")
    except Exception as e:
        print(f"Error setting seeds: {e}")
        # Fallback to default behavior if setting seeds fails

class CameraVideoMapper:
    """
    Manages mapping between video files and camera IPs
    Allows flexible configuration of camera-video associations
    """
    def __init__(self, mapping_config: Dict[str, str] = None):
        """
        Initialize the mapper with optional configuration
        
        :param mapping_config: Dictionary mapping video filenames to camera IPs
        """
        # Default mapping if not provided
        self.default_mapping = {
            'Accidents.mp4': '192.168.1.100',
            'Accident-1.mp4': '192.168.1.101',
            'Accident-2.mp4': '192.168.1.102'
        }
        
        # Override with provided mapping
        self.mapping = mapping_config or self.default_mapping
    
    def get_camera_ip(self, video_path: str) -> str:
        """
        Retrieve camera IP for a given video path
        
        :param video_path: Path to the video file
        :return: Corresponding camera IP
        """
        # Extract filename from full path
        filename = os.path.basename(video_path)
        
        # Lookup IP, fall back to default if not found
        camera_ip = self.mapping.get(filename, self.default_mapping.get(filename, '192.168.1.100'))
        
        logging.info(f"Video {filename} mapped to Camera IP: {camera_ip}")
        return camera_ip



class AccidentDetectionSystem:
    def __init__(self, camera_video_mapper: CameraVideoMapper, model_path: str = 'accident_detection_model.h5',seed=42):
        """
        Initialize Accident Detection System with optional SMS alerting capabilities
        
        :param camera_ips: List of IP addresses for cameras
        :param model_path: Path to save/load ML model
        """
        set_seeds(seed)
        self.camera_ips = []
        self.camera_video_mapper = camera_video_mapper
        self.model_path = model_path
        self.base_model = None
        self.model = None
        
        self._init_alert_config()
        self.load_or_create_model()

    def _init_alert_config(self):
        """
        Initialize alert configuration from environment variables
        Uses secure retrieval with fallback to prevent runtime errors
        """
        # Twilio Credentials
        self.twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
        self.twilio_from_number = os.getenv('TWILIO_FROM_NUMBER')
        self.control_room_number = os.getenv('CONTROL_ROOM_NUMBER')
        
        # Initialize Twilio client if all credentials are present
        self.twilio_client = None
        if all([self.twilio_account_sid, self.twilio_auth_token, 
                self.twilio_from_number, self.control_room_number]):
            try:
                from twilio.rest import Client
                self.twilio_client = Client(
                    self.twilio_account_sid, 
                    self.twilio_auth_token
                )
                logger.info("Twilio SMS alert system initialized")
            except ImportError:
                logger.error("Twilio library not installed. SMS alerts disabled.")
            except Exception as e:
                logger.error(f"Failed to initialize Twilio: {e}")

    def set_seeds(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)


    def send_sms_alert(self, message: str):
        """
        Send SMS alert using Twilio or fallback method
        
        :param message: Alert message to send
        """
        if self.twilio_client and self.control_room_number:
            try:
                message = self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_from_number,
                    to=self.control_room_number
                )
                logger.info(f"SMS Alert sent: {message.sid}")
            except Exception as e:
                logger.error(f"Failed to send Twilio SMS: {e}")
                self.send_alternative_alert(message)
        else:
            self.send_alternative_alert(message)

    def send_alternative_alert(self, message: str):
        """
        Alternative alert method using HTTP requests or other services
        
        :param message: Alert message to send
        """
        logger.warning("Sending alternative alert")
        try:
            # Example of using a simple HTTP POST request to a hypothetical alert service
            response = requests.post('https://alert-service.example.com/send', 
                                     json={
                                         'message': message, 
                                         'recipient': self.control_room_number
                                     })
            if response.status_code == 200:
                logger.info("Alternative alert sent successfully")
            else:
                logger.error(f"Alternative alert failed: {response.status_code}")
        except Exception as e:
            logger.error(f"Alternative alert method failed: {e}")

    def setup_logging(self):
        """Setup logging configuration"""
        logger.info("Initializing Accident Detection System")
        logger.info(f"Configured Cameras: {self.camera_ips}")

    def load_or_create_model(self):
        """Load existing model or create and save a new one"""
        if os.path.exists(self.model_path):
            logger.info(f"Loading existing model from {self.model_path}")
            self.model = load_model(self.model_path)
            
            # Recompile the model to ensure metrics are set up
            self.model.compile(
                loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy']
            )
            
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
            try:
                location = geocoder.ip(ip)
                if location.latlng:
                    try:
                        reverse_loc = geolocator.reverse(location.latlng)
                        camera_locations[ip] = reverse_loc.address if reverse_loc else "Location Unavailable"
                        logger.info(f"Camera IP {ip}: {camera_locations[ip]}")
                    except Exception as reverse_error:
                        logger.warning(f"Reverse geocoding failed for {ip}: {reverse_error}")
                        camera_locations[ip] = f"Approximate Location: Lat {location.latlng[0]}, Lon {location.latlng[1]}"
                else:
                    logger.warning(f"No location found for IP {ip}")
                    camera_locations[ip] = "Location Unknown"
            except Exception as e:
                logger.error(f"Geolocation lookup failed for {ip}: {e}")
                camera_locations[ip] = "Location Retrieval Failed"
        
        return camera_locations

    def detect_accidents(self, video_path: str) -> None:
        """Detect accidents in video stream"""
         # Dynamically get camera IP for this specific video
        camera_ip = self.camera_video_mapper.get_camera_ip(video_path)
        self.camera_ips = [camera_ip]  # Update camera IPs for this detection run
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
                    snapshot_path = os.path.join(AccSnapshotDir, snapshot_filename)
                    cv2.imwrite(snapshot_path, frame)
                    snapshot_counter += 1
                    imgflag = 1

                    # Send SMS Alert
                    try:
                        # Get location of camera that detected the accident
                        camera_location = self.get_geolocations().get(self.camera_ips[0], "Unknown Location")
                        
                        alert_message = f"""
ACCIDENT ALERT:
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Location: {camera_location}
Confidence: {percent:.2f}%
Snapshot: {snapshot_path}
"""
                        self.send_sms_alert(alert_message)
                    except Exception as e:
                        logger.error(f"Alert generation failed: {e}")


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
    
    # Create a custom mapping (optional)
    custom_video_ip_mapping = {
        'Accident-1.mp4': '1.39.116.199',
        'Accident-2.mp4': '1.38.116.199',
        # Add more mappings as needed
    }
    
     # Initialize the Camera Video Mapper
    camera_video_mapper = CameraVideoMapper(custom_video_ip_mapping)

    # Initialize system
    accident_system = AccidentDetectionSystem(camera_video_mapper,seed=42)
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