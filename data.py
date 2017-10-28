import csv
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

"""
Credit to NVIDIA and Former Udacity SDCND Student Jeff Wen(@jeffwen) for distribution technique
"""


def load_data_from_csv(data_dir):
    # Load Driving data
    samples = []
    with open(data_dir + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    samples = shuffle(samples)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    return map(np.array, [train_samples, validation_samples])


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype = np.float64)
    random_bright = .5+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1[:,:,2][image1[:,:,2]>255]  = 255
    image1 = np.array(image1, dtype = np.uint8)
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


def preprocess_image(image, color_conversion=cv2.COLOR_BGR2YUV):
    """
    Converts color of image and crops the top and bottom
    :param image:
    :param color_conversion:
    :return:
    """
    converted_image = cv2.cvtColor(image, color_conversion)

    cropped_image = converted_image[60:140, :, :]

    return cropped_image


def distribute_data(observations, min_needed=500, max_needed=750):
    """ create a relatively uniform distribution of images
    Arguments
        observations: the array of observation data that comes from the read input function
        min_needed: minimum number of observations needed per bin in the histogram of steering angles
        max_needed:: maximum number of observations needed per bin in the histogram of steering angles
    Returns
        observations_output: output of augmented data observations
    """

    observations_output = observations.copy()

    # create histogram to know what needs to be added
    steering_angles = np.asarray(observations_output[:, 3], dtype='float')
    num_hist, idx_hist = np.histogram(steering_angles, 20)

    to_be_added = np.empty([1, 7])
    to_be_deleted = np.empty([1, 1])

    for i in range(1, len(num_hist)):
        if num_hist[i - 1] < min_needed:

            # find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            # randomly choose up to the minimum needed
            need_to_add = observations_output[np.random.choice(match_idx, min_needed - num_hist[i - 1]), :]

            to_be_added = np.vstack((to_be_added, need_to_add))

        elif num_hist[i - 1] > max_needed:

            # find the index where values fall within the range
            match_idx = np.where((steering_angles >= idx_hist[i - 1]) & (steering_angles < idx_hist[i]))[0]

            # randomly choose up to the minimum needed
            to_be_deleted = np.append(to_be_deleted, np.random.choice(match_idx, num_hist[i - 1] - max_needed))

    # delete the randomly selected observations that are overrepresented and append the underrepresented ones
    observations_output = np.delete(observations_output, to_be_deleted, 0)
    observations_output = np.vstack((observations_output, to_be_added[1:, :]))

    return observations_output


def generate_data(data_dir, observations, batch_size=128):
    """ data generator in batches to be fed into the Keras fit_generator object
    Arguments
        observations: the array of observation data that is to be split into batches and read into image arrays
        batch_size: batches of images to be fed to Keras model
    Returns
        X: image array in batches as a list
        y: steering angle list
    """

    # applying correction to left and right steering angles
    steering_correction = 0.2

    # set up generator
    while True:
        for offset in range(0, len(observations), batch_size):
            batch_obs = shuffle(observations[offset:offset + batch_size])

            center_images = []
            left_images = []
            right_images = []

            steering_angle_center = []
            steering_angle_left = []
            steering_angle_right = []

            # loop through lines and append images + steering data to new lists
            for observation in batch_obs:
                center_image_path = data_dir + '/IMG/' + observation[0].split('/')[-1]
                left_image_path = data_dir + '/IMG/' + observation[1].split('/')[-1]
                right_image_path = data_dir + '/IMG/' + observation[2].split('/')[-1]

                center_images.append(preprocess_image(cv2.imread(center_image_path)))
                steering_angle_center.append(float(observation[3]))

                left_images.append(preprocess_image(cv2.imread(left_image_path)))
                right_images.append(preprocess_image(cv2.imread(right_image_path)))

                # append the steering angles and correct for left/right images
                steering_angle_left.append(float(observation[3]) + steering_correction)
                steering_angle_right.append(float(observation[3]) - steering_correction)

            images = center_images + left_images + right_images
            steering_angles = steering_angle_center + steering_angle_left + steering_angle_right

            X = np.array(images)
            y = np.array(steering_angles)

            yield shuffle(X, y)
