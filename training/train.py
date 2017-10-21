import csv 
import cv2
import numpy as np
import sys
import keras
import keras.models as models

from keras.models import load_model


# load the model
if len(sys.argv) != 5:
    print('Usage python train.py <path/to/driving/log/csv> <path/to/images/folder> <<path/to/existing/model> <path/to/new/model>')
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
    
    # Update path in log file to match the path on this system
    lines = lines[1:] # If first line is headings
    for line in lines:
    	source_path = line[0] # Center image 
    	filename = source_path.split('/')[-1]
    	current_path = img_folder_path + filename
    	image = cv2.imread(current_path)
    	images.append(image)
    	measurement = float(line[3]) 
    	measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)
    
    # Load the model 
    model = load_model(model_file_path)
    
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

    model.save(new_model_path)
