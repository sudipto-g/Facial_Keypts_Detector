[//]: # (Image References)

# Facial Keypoint Detection

This project aims to build a facial keypoint detection system.  
 
Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition.  


The project is broken up into a few main parts as follows:  

__Notebook 1__ : Loading and Visualizing the Facial Keypoint Data

__Notebook 2__ : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints

__predict.py__ : script to load image and detect keypoints in it




## Usage Instructions

1. Clone the repository, and navigate to the downloaded folder. 
```
git clone https://github.com/sudipto-g/Facial_Keypts_Detector.git
cd Facial_Keypts_Detector
```

2. Create (and activate) a new environment, named `key-pt`. If prompted to proceed with the install `(Proceed [y]/n)` type y.

	- __Linux__ or __Mac__: 
	```
	conda create -n key-pt
	source activate key-pt
	```
	- __Windows__: 
	```
	conda create --name key-pt
	activate key-pt
	```
	

3. Install PyTorch and torchvision; this should install the latest version of PyTorch.
	
	- __Linux__ or __Mac__: 
	```
	conda install pytorch torchvision -c pytorch 
	```
	- __Windows__: 
	```
	conda install pytorch-cpu -c pytorch
	pip install torchvision
	```

4. Install a few required pip packages, which are specified in the requirements text file (including OpenCV).
```
pip install -r requirements.txt
```

5. To run the facial keypoint detector on an image of your choice, place the image in the ```images/``` directory. And, then run the detector using
```
python predict.py name_of_file
```
Please note that while supplying the name of the image, you need to mention the extension too. For eg: ```python predict.py beatles.jpg``` is valid usage. 

## Data

All of the data needed to train a neural network is in the Facial\_Keypts\_Detector repo, in the subdirectory `data`.  
