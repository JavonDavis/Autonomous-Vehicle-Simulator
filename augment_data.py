import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import cv2

import numpy as np
import matplotlib.image as mpimg
import random

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def transform_image(img,ang_range,shear_range,trans_range,brightness=0):
    '''
    This function transforms images to generate new images.
    The function takes in following arguments,
    1- Image
    2- ang_range: Range of angles for rotation
    3- shear_range: Range of values to apply affine transform to
    4- trans_range: Range of values to apply translations over.

    A Random uniform distribution is used to generate different parameters for transformation

    '''
    # Rotation

    ang_rot = np.random.uniform(ang_range)-ang_range/2
    rows,cols,ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_rot,1)

    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    tr_y = trans_range*np.random.uniform()-trans_range/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])

    # Shear
    pts1 = np.float32([[5,5],[20,5],[5,20]])

    pt1 = 5+shear_range*np.random.uniform()-shear_range/2
    pt2 = 20+shear_range*np.random.uniform()-shear_range/2

    # Brightness


    pts2 = np.float32([[pt1,5],[pt2,pt1],[5,pt2]])

    shear_M = cv2.getAffineTransform(pts1,pts2)

    img = cv2.warpAffine(img,Rot_M,(cols,rows))
    img = cv2.warpAffine(img,Trans_M,(cols,rows))
    img = cv2.warpAffine(img,shear_M,(cols,rows))

    if brightness == 1:
      img = augment_brightness_camera_images(img)

    return img


def augment(features, labels, number_of_new_images):
    original_length = len(features)
    augmented_images = []
    augmented_images_labels = []
    previous_percentage = -1
    for count in range(number_of_new_images):
        index = random.randint(0, len(features))

        image = features[index]
        label = labels[index]

        augmented_image = transform_image(image, 20, 10, 5, brightness=1)
        augmented_images.append(augmented_image)

        features = np.concatenate((features, np.expand_dims(augmented_image, axis=0)))
        labels = np.append(labels, label)
        augmented_images_labels.append(label)
        current_percentage = int(count * 100 / float(number_of_new_images))
        if current_percentage > previous_percentage and current_percentage != 100:
            previous_percentage = current_percentage
            print('{}%'.format(current_percentage))
    assert (len(features) == original_length + number_of_new_images)
    print('100%')
    print("Added {} new images".format(number_of_new_images))
    return features, labels

if __name__ == '__main__':
    with open('train.p', mode='rb') as f:
        train = pickle.load(f)

    X_train, y_train = train['features'], train['labels']

    X_train, y_train = augment(X_train, y_train, 10000)
    augmented_data = {'features': X_train, 'labels':y_train}
    with open('train_augmented.p', mode='wb') as f:
        pickle.dump(augmented_data, f)
