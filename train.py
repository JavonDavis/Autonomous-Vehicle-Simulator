import csv
import cv2
import numpy as np
import sys
from augment_data import augment

from keras.models import load_model
from sklearn.utils import shuffle
import sklearn.metrics as metrics
import matplotlib.pyplot as plt


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]  # Center, left and right images image
                    filename = source_path.split('/')[-1]
                    current_path = img_folder_path + filename
                    image = cv2.imread(current_path)
                    images.append(image)
                    angle = float(batch_sample[3])
                    if i == 1:  # Left
                        angle += correction_factor
                    if i == 2:  # Right
                        angle -= correction_factor
                    angles.append(angle)
                    # Add augmented images by flipping horizontally and adjust steering angle
                    images.append(np.fliplr(image))
                    angles.append(-angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            # X_train, y_train = augment(X_train, y_train, 1000)  # Time consuming
            yield shuffle(X_train, y_train)


# load the model
if len(sys.argv) != 5:
    print(
        'Usage: python train.py'
        ' <path/to/driving/log/csv> <path/to/images/folder> <<path/to/existing/model> <path/to/new/model>')
else:

    driving_log_path, img_folder_path, model_file_path, new_model_path = sys.argv[1:]

    # Load Driving data
    samples = []
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    from sklearn.model_selection import train_test_split

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    correction_factor = 0.2

    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    # Load the model
    model = load_model(model_file_path)

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

    # print the keys contained in the history object
    print(history_object.history.keys())

    # plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    model.save(new_model_path)
