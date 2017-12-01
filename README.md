# Keras-tutorial-on-CNNs
We're going to build 3 image CNNs using just Tensorflow and Keras. The first is 
a simple classifier for images that will show you the basics of the keras api and how to build a
simple CNN. The second is a classifier that uses pre-trained VGG16 
convolutional layers and fine tunes them for a different classification task. The third model is a CNN regressor
taht also uses pre-trained VGG16 layers but fine tunes them for the task of face detection.
The goal for this is to fully understand how a Convolutional Neural Network works. 

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

- The simple CNN achieves approximately 80% accuracy on the cifar10 data after 50 epochs. 

- The CNN with the VGG16 pre-trained layers achieves approximately 91% accuracy on the cifar10 data.

- The CNN regressor achieves approximately 187 Mean Squared Error on the celebA data after 28 epochs. 