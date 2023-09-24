# %%
python pip install geopy

# %%
python pip install geocoder

# %%
python pip install scikit-image

# %%
python pip install scikit-learn

# %%
python pip install twilio

# %%
import os
import cv2     # for capturing videos
import math 
import geocoder
import requests
%matplotlib inline
import pandas as pd
from twilio.rest import Client
from geopy.geocoders import Nominatim
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from matplotlib import pyplot as plt 
from skimage.transform import resize   # for resizing images
import datetime
import base64

# %%
count = 0
videoFile = "Accidents.mp4"
cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path
frameRate = cap.get(5) #frame rate
x=1
path = "./traindata/"
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename =path+"%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

# %%
img = plt.imread('./traindata/0.jpg')   # reading image using its name
plt.imshow(img)

# %%
data = pd.read_csv('mapping.csv')     # reading the csv file
data.head()

# %%
X = [ ]     # creating an empty array
tpath="traindata/"
for img_name in data.Image_ID:
    img = plt.imread(tpath + img_name)
    X.append(img)  # storing each image in array X
X = np.array(X)    # converting list to array

# %%
y = data.Class
dummy_y = np_utils.to_categorical(y)

# %%
image = []
for i in range(0,X.shape[0]):
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    image.append(a)
X = np.array(image)

# %%
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X,data_format=None)

# %%
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)

# %%
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer, Dropout

# %%
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 

# %%
X_train = base_model.predict(X_train)
X_valid = base_model.predict(X_valid)
X_train.shape, X_valid.shape

# %%
X_train = X_train.reshape(155, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(67, 7*7*512)

# %%
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()

# %%
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation='sigmoid')) # hidden layer
model.add(Dense(2, activation='softmax'))    # output layer

# %%
model.summary()

# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))

# %%
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %%
count = 0
videoFile = "Accident-1.mp4"
cap = cv2.VideoCapture(videoFile)
frameRate = cap.get(5) #frame rate
x=1
while(cap.isOpened()):
    frameId = cap.get(1) #current frame number
    ret, frame = cap.read()
    path="./test/"
    if (ret != True):
        break
    if (frameId % math.floor(frameRate) == 0):
        filename =path+"test%d.jpg" % count;count+=1
        cv2.imwrite(filename, frame)
cap.release()
print ("Done!")

# %%
test = pd.read_csv('test.csv')

# %%
test_image = []
for img_name in test.Image_ID:
    img = plt.imread(path + img_name)
    test_image.append(img)
test_img = np.array(test_image)

# %%
test_image = []
for i in range(0,test_img.shape[0]):
    a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
    test_image.append(a)
test_image = np.array(test_image)

# %%
# preprocessing the images
test_image = preprocess_input(test_image, data_format=None)

# extracting features from the images using pretrained model
test_image = base_model.predict(test_image)
test_image.shape

# %%
test_image = test_image.reshape(9, 7*7*512)

# zero centered images
test_image = test_image/test_image.max()

# %%
predictions = model.predict(test_image)

# %%
print(predictions)

# %%
for i in range (0,9):
    if predictions[i][0]<predictions[i][1]:
        print("No Accident")
    else:
        print("Accident")
    

# %%
geoLoc = Nominatim(user_agent="GetLoc")
g = geocoder.ip('me')
locname = geoLoc.reverse(g.latlng)
account_sid = "ACc2a09b28a2a55c31129cb963048c4024"
auth_token = "72fa1731b0e3d58c8eab79830e178c49"
client = Client(account_sid, auth_token)

# %%
import cv2
cap = cv2.VideoCapture('Accident-2.mp4')
i=0
flag=0
snapshot_counter = 0
imgflag=0
while(True):
    ret,frame=cap.read()
    if ret==True:
        if predictions[int(i/15)%9][0]<predictions[int(i/15)%9][1]:
            percent = predictions[int(i/15)%9][1]*100
            predict="No Accident" #+ str(percent)
        else:
            percent = predictions[int(i/15)%9][0]*100
            predict="Accident " + str(percent)
            flag=1

            if imgflag==0 and percent >80:
                AccSnapshotDir = 'AccSnaps/'      
                snapshot_filename = f'accident_snapshot_{snapshot_counter}.jpg'
                cv2.imwrite(AccSnapshotDir + snapshot_filename, frame)
                snapshot_counter += 1
                imgflag=1

        # Save a snapshot at the time of accident
            
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame,
                predict,
                (50, 50),
                font, 1,
                (0, 255, 255),
                3,
                cv2.LINE_4)
        # cv2.putText(frame,
        #         percent,
        #         (0, 255, 255),
        #         font, 1,
        #         (0, 255, 255),
        #         3,
        #         cv2.LINE_4)
        cv2.imshow('Frame', frame)
        i=i+1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
if flag==1:
    
    with open(AccSnapshotDir + snapshot_filename, 'rb') as image_file:
        image_binary = image_file.read()
    base64_encoded = base64.b64encode(image_binary).decode('utf-8')
    # print(locname, datetime.datetime.now(), base64_encoded)
    client.messages.create(
                 body="Accident detected in "+locname.address,
                 from_= "+12568040182",
                 to= "+919074062399"
                 )
    # print(str(datetime.datetime.now().strftime("%X")))
    data = {
    "_loc": str(locname),
    "_time": str(datetime.datetime.now().strftime("%X")),
    "_date": str(datetime.datetime.now().strftime("%x")),
    "_plate" : "MH 12 1234",
    "snapShot": base64_encoded
    } 
    # files = {'_snapShot': open(AccSnapshotDir + snapshot_filename, 'rb')}
    response = requests.post('http://localhost:3000/addAccident', json=data)
    if response.status_code == 200:
        print("POST request successful")
    else:
        print("POST request failed with status code:", response.status_code)
        print("Response content:", response.text)

# release the cap object
cap.release()
# close all windows
cv2.destroyAllWindows()



