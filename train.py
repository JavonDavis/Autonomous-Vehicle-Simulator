import csv
import cv2
import numpy as np
import sys
from augment_data import augment

from keras.models import load_model
from sklearn.utils import shuffle

# load the model
if len(sys.argv) != 5:
    print(
        'Usage: python train.py'
        ' <path/to/driving/log/csv> <path/to/images/folder> <<path/to/existing/model> <path/to/new/model>')
else:

    driving_log_path, img_folder_path, model_file_path, new_model_path = sys.argv[1:]

    # Load Driving data
    lines = []
    with open(driving_log_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    images = []
    measurements = []

    correction_factor = 0.2

    # Update path in log file to match the path on this system
    # lines = lines[1:] # If first line is headings
    for line in lines:
        for i in range(3):
            source_path = line[i]  # Center, left and right images image
            filename = source_path.split('/')[-1]
            current_path = img_folder_path + filename
            image = cv2.imread(current_path)
            images.append(image)
            measurement = float(line[3])
            if i == 1:  # Left
                measurement += correction_factor
            if i == 2:  # Right
                measurement -= correction_factor
            measurements.append(measurement)
            # Add augmented images by flipping horizontally and adjust steering angle
            images.append(np.fliplr(image))
            measurements.append(-measurement)

    X_train, y_train = np.array(images), np.array(measurements)
    # X_train, y_train = augment(X_train, y_train, 10000) # Time consuming
    X_train, y_train = shuffle(X_train, y_train)

    # Load the model
    model = load_model(model_file_path)

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

    model.save(new_model_path)
