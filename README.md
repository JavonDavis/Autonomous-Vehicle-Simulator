# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---

In this project, I used a convolutional neural network to clone driving behavior. I trained, validated and tested a model using Keras. The model outputs a steering angle to an autonomous vehicle.

A simulator is provided where you can steer a car around a track for data collection. I used image data and steering angles to train the neural network, the model was continually improved using the transfer learning technique and then used to drive the car autonomously around the track.

See below for links to download the Simulator:

Linux: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip

OSX: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip

Windows: https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip

# Final Model
Many iterations of the model can be found in the model folder but the final model is in the root of the directory and is named 'model.h5'

## Executing the final model
The final model has not been optimized for handling the second track as yet so YMMV when driving on that track. To execute simply:
1. run python drive.py model.h5 from the root directory
2. Fire up the simulator
3. Click Autonomous

A video of the final model doing 1 lap around the track can also be watched, see video.mp4 file.

## Model Architecture

The model architecture chosen is a slight variation based on [NVIDIA's End to End Learning for Self-Driving Cars paper](https://arxiv.org/pdf/1604.07316v1.pdf), the normalization laryer is a Lambda layer that applies a simple function to each pixel in an image(model.py line 32).


The network consists of 9 layers, including a normalization layer, 5 convolutional layers and 3 fully connected layers. The input image is split into YUV planes and passed to the network.

Before testing the model I wanted to put measures in place to prevent the network from Overfitting, I had initially attempted adding a Dropout layer after each convolutional layer but results were not optimal. After a bit of experimentation and googling I decided a single Dropout layout layer with 50% Keep probability immediately before the Fully connected layers(model.py line 41) would suffice.  

Original NVIDIA Architecture

![Original NVIDIA Model](/readme_images/nvidia.png?raw=true "Original NVIDIA Model")

Summary of the model used in this project
![Model Summary](/readme_images/model_summary.png?raw=true "Model Summary") (Lines 32 - 46)

### Model parameter tuning
The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 53)

### Training data
Thanks to the simulator training data was easy to gather via training mode. The data comprised of 3 camera angles, from the front of the car(from the center, left and right). See below for a sample,

Left
![Left Sample](/readme_images/left_sample.jpg?raw=true "Left Sample")

Center
![Center Sample](/readme_images/center_sample.jpg?raw=true "Center Sample")

Right
![Right Sample](/readme_images/right_sample.jpg?raw=true "Right Sample")

The final model was trained with 2 laps both clockwise and counter clockwise going around the first track.

## Solution Design
My first step was to start with the vanilla NVIDIA architecture with data from a single lap around track 1. The model drove the car a little but didn't make it pass the first corner. A data preprocessing step was introduced after collection with data which involved aligning the color space used for driving the car and training the network (data.py lines60-71) and correcting the steering angle for the left and right camera images with a correction factor of 0.2 (data.py lines 158 and 159).The data was the shuffled and 20% of the data was put into a validation set.

Research online then led me to realize that the data gathered is skewed towards driving straight(steering angle = 0) so then I us an implementation found online that distributes the data across 'buckets' based on the steering angle(data.py line 74 -116) so that the result is a more uniform dataset.  

After integrating this into the data preprocessing step the car actually made it (with a few bumps) to the first sharp right turn. As expected the car went straight into the water. This was expected because I was aware that the model had not gotten enough training with handling right turns since the track itself is biased towards left turns. The model was then training with 2 laps counter clockwise around the track and one more clockwise lap. See below for the validation loss achieved by the model at each stage. 

1 clockwise
![1 clockwise](/readme_images/Figure_1.png?raw=true "1 clockwise")

1 counter-clockwise
![1 counter-clockwise](/readme_images/Figure_2.png?raw=true "1 counter-clockwise")

2 counter-clockwise
![2 counter-clockwise](/readme_images/Figure_3.png?raw=true "2 counter-clockwise")

2 clockwise
![2 clockwise](/readme_images/Figure_4.png?raw=true "2 clockwise")

After each training session using 5 epochs each the car drove better each time until now it is succesfully making it around the track.
