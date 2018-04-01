# Motion Classifier
The goal of this project is to classify moving objects in a video. Each frame of the video is analyzed using OpenCV to find moving objects within the frame.  Each moving object is fed into a CNN that classifies the object and a bounding box is put around the object.   
   
This approach would work well on security cameras. It could log information on an object entering and leaving the frame which would allow for passive monitoring.   

## Transfer learning
The CNN was build on top of a pretrained network to reduce training time and increase accuracy given the limited training set. The VGG19 network was reconstructed and pretrained weights were loaded into the network. These layers were then frozen and custom layers were added to the network.

## Finding object
Moving objects are found in the frame using *cv2.createBackgroundSubtractorMOG2()*. Using this, contours of the moving objects are found and a bounding box for the contours is created. The coordinates of the bounding box are used to crop the frame which is then fed into the CNN
   