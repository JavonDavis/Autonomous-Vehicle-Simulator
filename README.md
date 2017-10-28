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
Many iterations of the model can be found in the model folder 

## Model Architecture
A basic model was built using model.py.
![Model Summary](/readme_images/model_summary.png?raw=true "Model Summary") (Lines 18 - 34)

The model was then improved using Transfer learning withthe train.py file.

The architecture I chose is based on NVIDIA's autopilot architecture but the Normalization layer was replaced with a Lambda layer that applies a simple function to each pixel in an image(pixel/255 - 0.5) and to combat Overfitting a Dropout layer was introduced after each Convolutional layer.

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs I went with was 2 even though I could possibly train for more since the validation loss was decreasing but Hardware limitations again made this difficult for me to play with and I also used an adam optimizer so that manually training the learning rate wasn't necessary.

## Model training
The final model used was trained with data from the 3 camera angles, and also flipped horizontally. This data was recorded from 3 laps - 2 clockwise laps and 1 counter clockwise lap on track1. I think more Data augmentation techniques could be employed to allow the model to perform better on both tracks but Hardware limitations made that difficult to accomplish for this submission.I realized that on both tracks the model(using models/model_06 for track2 which was trained with data from track2) didn't perform well on areas of the track that were darker(due to a shadow). I believe this could be definately be combatted with data augmentation techniques.
