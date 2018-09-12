# Ship Detection in Satelite Images
In this repo I present my solution for the ![Airbus Ship Detection challenge](https://www.kaggle.com/c/airbus-ship-detection) on Kaggle.
There was a data leakage in this competition which allowed for almost perfect scores. However, I used this comptetion as an introduction to computer vision and learned a lot about neural networks. 

## Approach
I developed two models. The fist is a quite simple image classification model which predicts if there is a boat within the image or not. The second is a sematic segmentation model which divides the image into background and foreground (=ships). Both neural networks use the Keras library

## Image classification model
For the image classification model I have trained a simple Convolution Neural Network (CNN). It takes an input of 512 x 512 x 3 image and generates an output of (None, 1). 

A visualization of the network can be found here: 

*Put tensorflow graph generation*

