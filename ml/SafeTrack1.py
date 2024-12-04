import os
import cv2
import math
import geocoder
import requests
import pandas as pd
import numpy as np
from twilio.rest import Client
from geopy.geocoders import Nominatim
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.layers import Dense, InputLayer, Dropout
from matplotlib import pyplot as plt
from skimage.transform import resize
import datetime
import base64

class AccidentDetectionSystem:
    def __init__(self, model_path='accident_detection_model.h5'):
        """Initialize the accident detection system."""
        self.model_path = model_path
        self.model = None
        self.base_model = None
        self.frame_size = (224, 224)
        
        # Configuration
        # self.twilio_account_sid = "YOUR_ACCOUNT_SID"
        # self.twilio_auth_token = "YOUR_AUTH_TOKEN"
        # self.twilio_from_number = "YOUR_TWILIO_NUMBER"
        # self.twilio_to_number = "YOUR_TARGET_NUMBER"
        
        # Initialize paths
        self.train_data_path = "./traindata/"
        self.test_data_path = "./test/"
        self.snapshot_path = "./AccSnaps/"
        
        # Create necessary directories
        os.makedirs(self.train_data_path, exist_ok=True)
        os.makedirs(self.test_data_path, exist_ok=True)
        os.makedirs(self.snapshot_path, exist_ok=True)

    def extract_frames(self, video_path, output_path, prefix="frame"):
        """Extract frames from a video file."""
        count = 0
        cap = cv2.VideoCapture(video_path)
        frame_rate = cap.get(5)
        
        while cap.isOpened():
            frame_id = cap.get(1)
            ret, frame = cap.read()
            if not ret:
                break
            if frame_id % math.floor(frame_rate) == 0:
                filename = f"{output_path}/{prefix}{count}.jpg"
                cv2.imwrite(filename, frame)
                count += 1
        
        cap.release()
        return count

    def prepare_data(self, data_csv):
        """Prepare training or testing data."""
        data = pd.read_csv(data_csv)
        images = []
        for img_name in data.Image_ID:
            img = plt.imread(self.train_data_path + img_name)
            resized_img = resize(img, preserve_range=True, output_shape=self.frame_size).astype(int)
            images.append(resized_img)
        
        X = np.array(images)
        X = preprocess_input(X, data_format=None)
        
        if 'Class' in data.columns:  # Training data
            y = np_utils.to_categorical(data.Class)
            return X, y
        return X

    def build_model(self):
        """Build and compile the model."""
        if os.path.exists(self.model_path):
            print("Loading existing model...")
            self.model = load_model(self.model_path)
            return

        print("Building new model...")
        self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*self.frame_size, 3))
        
        # Create and compile model
        self.model = Sequential([
            InputLayer((7*7*512,)),
            Dense(1024, activation='sigmoid'),
            Dropout(0.5),
            Dense(2, activation='softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, training_csv, epochs=100):
        """Train the model using the provided data."""
        X, y = self.prepare_data(training_csv)
        
        # Get features from base model
        X = self.base_model.predict(X)
        X = X.reshape(X.shape[0], -1)
        X = X / X.max()
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Train model
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid))
        
        # Save model
        self.model.save(self.model_path)

    def process_video(self, video_path, threshold=0.8):
        """Process video for accident detection."""
        cap = cv2.VideoCapture(video_path)
        frame_buffer = []
        predictions_buffer = []
        snapshot_counter = 0
        accident_detected = False
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            processed_frame = resize(frame, preserve_range=True, output_shape=self.frame_size).astype(int)
            frame_buffer.append(processed_frame)
            
            # Make prediction when buffer is full
            if len(frame_buffer) == 15:
                X = np.array(frame_buffer)
                X = preprocess_input(X, data_format=None)
                X = self.base_model.predict(X)
                X = X.reshape(X.shape[0], -1)
                X = X / X.max()
                
                pred = self.model.predict(X)
                pred_avg = np.mean(pred, axis=0)
                
                # Check for accident
                if pred_avg[1] > threshold and not accident_detected:
                    accident_detected = True
                    # Save snapshot
                    snapshot_path = f"{self.snapshot_path}/accident_snapshot_{snapshot_counter}.jpg"
                    cv2.imwrite(snapshot_path, frame)
                    snapshot_counter += 1
                    
                    # Send alert
                    #######self.send_alert(snapshot_path)
                
                # Display prediction
                text = f"Accident: {pred_avg[1]*100:.2f}%" if pred_avg[1] > threshold else "No Accident"
                cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                
                # Clear buffer
                frame_buffer = []
                
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
        return accident_detected

   ####### def send_alert(self, snapshot_path):
        """Send alert when accident is detected."""
        try:
            # Get location
            geoLoc = Nominatim(user_agent="GetLoc")
            g = geocoder.ip('me')
            locname = geoLoc.reverse(g.latlng)
            
            # Prepare image
            with open(snapshot_path, 'rb') as image_file:
                base64_encoded = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare alert data
            alert_data = {
                "_loc": str(locname),
                "_time": datetime.datetime.now().strftime("%X"),
                "_date": datetime.datetime.now().strftime("%x"),
                "_plate": "MH 12 1234",  # This should be replaced with actual plate detection
                "snapShot": base64_encoded
            }
            
            # Send to server
            response = requests.post('http://localhost:3000/addAccident', json=alert_data)
            
            # Send SMS
            client = Client(self.twilio_account_sid, self.twilio_auth_token)
            client.messages.create(
                body=f"Accident detected at {locname.address}",
                from_=self.twilio_from_number,
                to=self.twilio_to_number
            )
            
            print("Alert sent successfully")
            
        except Exception as e:
            print(f"Error sending alert: {str(e)}")

def main():
    # Initialize system
    system = AccidentDetectionSystem()
    
    # Build/load model
    system.build_model()
    
    # If training is needed
    if not os.path.exists(system.model_path):
        print("Extracting training frames...")
        system.extract_frames("Accidents.mp4", system.train_data_path)
        print("Training model...")
        system.train_model("mapping.csv")
    
    # Process test videos
    test_videos = ["Accident-1.mp4", "Accident-2.mp4"]
    for video in test_videos:
        if os.path.exists(video):
            print(f"Processing {video}...")
            accident_detected = system.process_video(video)
            if accident_detected:
                print(f"Accident detected in {video}")

if __name__ == "__main__":
    main()