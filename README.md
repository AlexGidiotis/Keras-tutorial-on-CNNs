# Keras-tutorial-on-CNNs
Tutorial on how to implement a simple CNN for image classification and face detection.

## Overview ##

This is the code for the workshop "Implementing CNNs with keras and tensorflow".

Includes 3 different CNN architectures:

1) **A simple CNN for image classification**

2) **A CNN that uses the pretrained layers from VGG16 for image classification**

3) **A CNN regressor that uses the pretrained layers from VGG16 for the task of face detection**

## Requirements

- Python
- NumPy
- OpenCV
- scikit-learn
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Keras](https://github.com/fchollet/keras)

## Usage

### Training
**Step 1.** 
Clone this repository with ``git``.

```
$ git clone https://github.com/AlexGidiotis/Keras-tutorial-on-CNNs.git
$ cd Keras-tutorial-on-CNNs
```

**Step 2.** 
Download [celebA data](https://www.dropbox.com/sh/hx19bwxdpn8xv33/AABkCRUPwfFi0xqcvXjMO8GFa?dl=0)

```
$ mkdir data/
```

**Step 3.** 
Move the data you downloaded to the data directory you just created.


**Step 4.** 
Created a directory to save the trained models.

```
$ mkdir model/
```


**Step 5.**
Try training your own models.

## Results ##

- The simple CNN achieves approximately 73% accuracy on the cifar10 data after 43 epochs. 

- The CNN with the VGG16 pre-trained layers achieves approximately 91% accuracy on the cifar10 data.

- The CNN regressor achieves approximately 187 Mean Squared Error on the celebA data after 28 epochs. 