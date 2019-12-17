import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

data = np.load('datafile.npy')
labels = np.load('labelfile.npy')

(trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.15, random_state=42)

# create model
model = Sequential()

# add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28 ,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(36, activation='softmax'))

# compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(trainData, trainLabels, validation_data=(testData, testLabels), epochs=10)
