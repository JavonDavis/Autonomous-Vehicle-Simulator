from keras.layers import Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

nrows = 160
ncols = 320

model = Sequential()

model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(nrows, ncols, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.2))
model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='tanh'))

model.summary()
# Save model to JSON
model.save('models/basic_model.h5')