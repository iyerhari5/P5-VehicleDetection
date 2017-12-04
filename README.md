# Vehicle Detection

The goals / steps of this project are the following:

* Perform feature extraction on a labeled training set of images and train a  Linear SVM classifier
* Train the classifier to disinguish between car and non-car images
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run vehicle detection pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[image1]: ./output_images/hog.jpg
[image2]: ./output_images/test-images-detection-1.png
[image3]: ./output_images/test-images-detection-2.png
[image4]: ./output_images/test-images-detection-3.png
[image5]: ./output_images/test-images-detection-4.png
[image6]: ./output_images/heat-map.png
[image7]: ./output_images/output-video.gif
[image8]: ./output_images/output-video-with-lanes.gif
[image9]: ./output_images/car.jpg


## Feature Extraction

In order to train the classifier, we first need to extract some features from the images. In this project we use the Histogram of Oriented Gradients (HOG),  binned color, and color histogram features. The HOG feature try to capture the edge information from the image that essentially encodes the shape of the object we are triyng to classify. The binned color image encodes both the color and shape of the object. The color histogram feature tries to capture the color information of the objects being classified. Figure below shows a sample image and the extracted HOG features.

![alt text][image9]
![alt text][image1]

After experimenting with different color transforms and spatial parameters, the final parameters used for the feature extraction are:

```
color_space = 'YCrCb' 
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

## Classifier Training

We use a linear SVM (Support Vector Machine) classifier. The classifier is trained with a set of labeled training images that are part of car and non-car classes. The features are extracted for each image and the features are normalized using StandardScaler() from sklearn package. The data is split into training and test datasets using the train_test_split() function with a 80%-20% split. The classifier obtained an accuracy of 98.7% on the test set. The output from the training is:

```
Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
12.49 Seconds to train SVC...
Test Accuracy of SVC =  0.987
```


## Sliding window based detection

The idea here is to slide a window over the image and to classify each window individually. Figures below show sample detections on the test images with a sliding window approach.

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

Due to the perspective distortion, cars that are further away from the camera will appear smaller  and cars that are near to the camera will appear bigger. Hence we need to adjust the size of the window used to detect the car based on the distance from the camera. The scale of the search window is hence controlled based on the y co-ordinates of the image region. Images towards the top of the image are further away from the camera and vice versa. The scales and y limits used are:

```
ystart = [400,  400, 400 ]
ystop =  [480,  500, 650 ]
scales = [1.0,  1.5, 2.0 ]
```


## Outlier rejection using heat maps

The above scheme of detection results in many outlier detections. In order to get more confidence in the detection, we use a heat map approach. The idea here is to create a heat map where the pixels with multiple detected boxes over it have higher intensities. The heat map is then thresholded to find the pixels that have high confidence if being a car. Figure below shows the individual box detections, the heat map and the final car locations obtained from the thresholded heat map.

![alt text][image6]

In the video pipeline, we integrate the heat map over a number of frames.

## Video Implementation

The results on the project video are shown below:
![alt text][image7]


## Simultaneous lane and vehicle detection

In this, [lane detection](https://github.com/iyerhari5/P4-AdvancedLaneFinding) is combined with the vehicle detection. The resulting video is shown below:
![alt text][image8]

## Potential improvements

The current pipeline is still not very robust and detects a few outliers.  One idea is to restrict the region of search in some frames based on prior detection. We could also smooth the bounding boxes across frames to get a more stable detection. The current pipeline also has no inforation on individual vehicles. A potential improvement is to track individual detections over frames.
