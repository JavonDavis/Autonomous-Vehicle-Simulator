import sys
from keras.layers import Lambda, BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from data import load_data_from_csv, generate_data
import matplotlib.pyplot as plt

nrows = 80
ncols = 320

# load the model
if len(sys.argv) != 3:
    print(
        'Usage: python model.py'
        ' <path/to/data/dir>  <path/to/model>')
else:

    data_dir, model_path = sys.argv[1:]

    train_samples, validation_samples = load_data_from_csv(data_dir)

    # compile and train the model using the generator function
    train_generator = generate_data(data_dir, train_samples)
    validation_generator = generate_data(data_dir, validation_samples)

    model = Sequential()

    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(nrows, ncols, 3)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
    # model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(3, nrows,ncols)))

    model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    checkpoint = ModelCheckpoint(model_path, verbose=0, save_best_only=True)
    callbacks_list = [checkpoint]

    model.summary()

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples),
                                         validation_data=validation_generator,
                                         nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1,
                                         callbacks=callbacks_list)

    # model.save(model_path)
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
