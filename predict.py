#########path to image - change this to get predictions on different image##########
image_path = 'images/beatles.jpg'
####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_all_keypoints(image, predicted_key_pts):
    """To show image with predicted keypoints"""
    # image is grayscale
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=5, marker='.', c='m')
    plt.title('Detected KeyPoints')

# loading in color image for face detection
image = cv2.imread(image_path)

# switching red and blue color channels 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(image)
plt.title('Original Image')
plt.show()

# loading in a haar cascade classifier for detecting frontal faces
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')

# running the detector
# the output here is an array of detections; the corners of each detection box
faces = face_cascade.detectMultiScale(image, 1.2, 2)
image_with_detections = image.copy()

for (x,y,w,h) in faces:
    # drawing a rectangle around each detected face
    # the width of the rectangle may need to be changed depending on image resolution
    cv2.rectangle(image_with_detections,(x,y),(x+w,y+h),(255,0,0),3) 

plt.figure()
plt.imshow(image_with_detections)
plt.title('Detected Face(s)')
plt.show()

import numpy as np

import torch
from models import Net

net = Net()

net.load_state_dict(torch.load('keypoints_model.pt'))

image_copy = np.copy(image)

# looping over the detected faces from haar cascade
for (x,y,w,h) in faces:
    
    # Selecting the region of interest that is the face in the image 
    roi = image_copy[y:y+h, x:x+w] # the numeric values are needed for scaling the output keypoints correctly.
    
    width_roi = roi.shape[1] # needed later for scaling keypoints
    height_roi = roi.shape[0] # needed later for scaling keypoints
    
    roi_copy = np.copy(roi) # will be used as background to display final keypints.
    
    ## Converting the face region from RGB to grayscale
    roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

    ## Normalizing the grayscale image so that its color range falls in [0,1] instead of [0,255]
    roi = roi/255.0
    
    ## Rescaling the detected face to be the expected square size for your CNN (224x224, suggested)
    roi = cv2.resize(roi, (96, 96))  # resizing the image to get a square 96*96 image.
    roi = np.reshape(roi,(96, 96, 1)) # reshaping after rescaling to add the third color dimension.
    
    ## Reshaping the numpy image shape (H x W x C) into a torch image shape (C x H x W)
    roi = roi.transpose(2, 0, 1)
    
    ## Makeing facial keypoint predictions using loaded, trained network 
    roi = torch.from_numpy(roi).type(torch.FloatTensor) # converting images to FloatTensors (common source of error)
    
    # Runtime error : expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, 
    # but got stride=[1, 1]. This error will occur if the following line is omitted. Hence we need to apply unsqueeze operation.
    roi = roi.unsqueeze(0)
    
    # Passing the transformed input images to the network for detecting keypoints.
    keypoints = net(roi)
    keypoints = keypoints.view(68, 2)
    
    # Undoing the transformations performed on the facial keypoints
    keypoints = keypoints.data.numpy()
    keypoints = keypoints*50.0 + 100
    
    ## Displaying each detected face and the corresponding keypoints        
    keypoints = keypoints * (width_roi / 96, height_roi / 90) # scaling the keypoints to match the size of the output display. 
     
    # Using helper function for display as defined previously.  
    show_all_keypoints(roi_copy, keypoints)
    plt.show()
