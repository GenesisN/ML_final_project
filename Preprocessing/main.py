from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
from tempfile import TemporaryFile
from keras.layers import Activation
from keras.models import Sequential
from keras.optimizers import SGD, Nadam
from keras.utils import np_utils
from keras.layers import Dense, Dropout
from keras.models import save_model
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import imutils
import math
import sys
import os


def create_model(learn_rate=0.01):
    # create model
    model = Sequential()
    model.add(Dense(11, kernel_initializer="uniform"))
    model.add(Dense(29, activation="relu", kernel_initializer="uniform"))
    model.add(Dense(36))
    model.add(Activation("softmax"))
    optimizer = Nadam(lr=learn_rate)
    # Compile model
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model

seed = 7
np.random.seed(seed)

labelfile = TemporaryFile()
datafile = TemporaryFile()


def crop_image(binary_image, y, x, w, h):
    crop = binary_image[y:y + h, x:x + w]
    return crop


def check_right_wrist(binary_image, y, w, h):
    right_pixels = 0
    status = False

    for i in range(y, h):
        right_pixel_val = binary_image[i][w-1]
        if right_pixels == 15:
            status = True
            break

        if right_pixel_val == 255:
            right_pixels += 1
        else:
            right_pixels = 0

    return status


def check_top_wrist(binary_image, x, w):
    top_pixels = 0
    status = False

    for i in range(x, w):
        top_pixel_val = binary_image[0][i]
        if top_pixels == 15:
            status = True
            break

        if top_pixel_val == 255:
            top_pixels += 1
        else:
            top_pixels = 0

    return status


def check_bottom_wrist(binary_image, w, h):
    bottom_pixels = 0
    status = False

    for i in range(w):
        bottom_pixel_val = binary_image[h - 1][i]
        if bottom_pixels == 15:
            status = True
            break  # don't rotate

        if bottom_pixel_val == 255:
            bottom_pixels += 1
        else:
            bottom_pixels = 0
    # if counter not 15, check right and top borders
    return status


# Function to check for a wrist at the bottom, top and right borders
def find_wrist(binary_image, x, y, w, h):
    if check_bottom_wrist(binary_image, w, h):
        return binary_image

    elif check_top_wrist(binary_image, x, w):
        erosion = imutils.rotate(binary_image, 180)
        return erosion

    elif check_right_wrist(binary_image, y, w, h):
        erosion = imutils.rotate(binary_image, 270)
        return erosion

    else:
        return binary_image


def find_finger_tip(binary_image, x, y, w, h):
    #print(x, y, w, h)
    x_coords = []
    y_coords = []
    finger_count = 0
    pixel_counter = 0

    # iterate over the image in x and y
    for j in range(y, h):
        for i in range(x, w):
            pixel = binary_image[j][i]
            # is the pixel  white ?
            if pixel == 255:
                pixel_counter += 1
                # white pixel counter checks for 6 consecutive white pixels
                if pixel_counter == 6:
                    if check_left_right(binary_image, i, pixel_counter, j, w):
                        if j == y:
                            if check_bottom(binary_image, i, j, pixel_counter):
                                x_coords.append(i)
                                y_coords.append(j)
                                finger_count += 1
                                if finger_count == 5:
                                    break
                            else:
                                pixel_counter = 0
                        else:
                            if check_top_bottom(binary_image, i, j, pixel_counter, w, h):
                                x_coords.append(i)
                                y_coords.append(j)
                                finger_count += 1
                                if finger_count == 5:
                                    break
                            else:
                                pixel_counter = 0
                    else:
                        pixel_counter = 0
            else:
                pixel_counter = 0
    if finger_count == 0:
        x_coords = [1.0]*5
        y_coords = [1.0]*5
    return [finger_count, x_coords, y_coords]


# check for left and right of three continuous pixels
def check_left_right(binary_image, i, p_counter, j, w):
    status = False
    left_pixel = i - p_counter

    right_pixel = i + 1
    if right_pixel < w:
        if binary_image[j][right_pixel] == 0 and binary_image[j][left_pixel] == 0:
            status = True
    else:
        status = False
    return status


# checking bottom if on top row
def check_bottom(binary_image, i_val, j_val, counter):
    bottom_row = j_val + 1
    status = False
    start = i_val - (counter - 1)
    end = i_val + 1

    for i in range(start, end):
        if binary_image[bottom_row][i] == 255:
            status = True
        else:
            status = False
            break
    return status


# check if top pixels are black or single white pixel
# and bottom pixels are white
def check_top_bottom(binary_image, i_val, j_val, counter, w, h):
    bottom_row = j_val + 1
    top_row = j_val - 1
    if bottom_row >= h:
        bottom_row = h-1
        #print("hhhhhhh",h)

    start = i_val - (counter - 1)
    end = i_val + 1
    if end >= w:
        end = w

    bottom_status = False
    top_status = False
    general_status = False

    # checking top rows
    white_pixel = 0
    for i in range(start, end):
        if binary_image[top_row][i] == 0:
            top_status = True
        else:
            white_pixel += 1
            top_status = False

    if 1 <= white_pixel <= 2 or top_status == True:
        top_status = True

    # checking bottom rows
    for i in range(start, end):
        #print(start, end)
        #print(bottom_row,i)
        if binary_image[bottom_row][i] == 255:
            bottom_status = True
        else:
            bottom_status = False
            break

    # checking to ensure that a finger tip has been found
    if bottom_status == True and top_status == True:
        general_status = True

    return general_status


def find_centroid(x_coords, y_coords):
    n = len(x_coords)
    x_coord = sum(x_coords) / n
    y_coord = sum(y_coords) / n

    centroid = x_coord, y_coord

    return list(centroid)


def calc_angle_distance(centroid, vertice_x, vertice_y):
    angle_distance_list = []

    for i in range(len(vertice_x)):
        dx = centroid[0] - vertice_x[i]
        dy = centroid[1] - vertice_y[i]
        rad = math.atan2(dy, dx)
        deg = math.degrees(rad)
        angle_distance_list.append(deg)

        distance = math.sqrt(((vertice_x[i] - centroid[0]) ** 2) + (vertice_y[i] - centroid[1]) ** 2)
        angle_distance_list.append(distance)

    return angle_distance_list


def calc_area(binary_image, x, y, w, h):
    white_pixels = 0
    for i in range(x, w):
        for j in range(y, h):
            if binary_image[i][j] == 255:
                white_pixels += 1
            else:
                continue

    return white_pixels


def pre_process_image(image_name):
    image = cv.imread(image_name)
    #print(image_name)
    image = cv.resize(image, (260, 260))
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(image, (5, 5), 0)
    ret, binary_img = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(binary_img, kernel, iterations=1)

    contours, _ = cv.findContours(erosion, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv.boundingRect(c)
        if rect[2] < 100 or rect[3] < 100:
            continue
        x, y, w, h = rect

    return [erosion, x, y, w, h]


def feature_extraction(image_name):
    pre_processed_image = pre_process_image(image_name)

    rotated_image = find_wrist(pre_processed_image[0], pre_processed_image[1],
               pre_processed_image[2], pre_processed_image[3],
               pre_processed_image[4])

    features = find_finger_tip(rotated_image, pre_processed_image[1],
               pre_processed_image[2], pre_processed_image[3],
               pre_processed_image[4])

    centroid = find_centroid(features[1], features[2])

    angles = calc_angle_distance(centroid, features[1], features[2])

    area = calc_area(pre_processed_image[0], pre_processed_image[1],
               pre_processed_image[2], pre_processed_image[3],
               pre_processed_image[4])

    angles.append(area)

    # format to make sure we have a vector at the end of the day
    #print("features", angles)
    return angles

def main(folder_path):
    # list storing image labels
    data = []
    labels = []
    vec = []
    j = 1
    # looping through list of all image file names
    # for file_name in os.listdir(folder_path):
    #     label = file_name.index("_") + 1
    #     if file_name != ".DS_Store":
    #         features = feature_extraction(folder_path + "/" + file_name)
    #
    #         if len(features) < 11:
    #             for i in range(len(features), 11):
    #                 features.append(0.0)
    #
    #         if len(features) > 11:
    #             area = features[-1]
    #             features = features[0:10]
    #             features.append(area)
    #
    #         #print(str(j) + " " + str(len(features)))
    #
    #         # adding extracted label and filename from image to the list of data and labels
    #         data.append(features)
    #         labels.append(file_name[label])
    #         j += 1
    #         if j > 0 and j % 1000 == 0:
    #             print("[INFO] processed {}/{}".format(j, len(os.listdir(folder_path))))
    #     else:
    #         continue
    # vec = vec + labels
    # vec = list(set(vec))
    # #print(vec)
    # le = LabelEncoder()
    # labels = le.fit_transform(labels)
    #
    # data = np.array(data) / 255.0
    # labels = np_utils.to_categorical(labels, 36)

    # print("[INFO] saving labels and data...")
    # np.save('labelfile.npy', labels)
    # np.save('datafile.npy', data)

    print("[INFO] loading data")
    data = np.load('../datafile.npy')
    labels = np.load('../labelfile.npy')

    print("[INFO] splitting data")
    (trainData, testData, trainLabels, testLabels) = train_test_split(data, labels, test_size=0.15, random_state=42)

    # Grid search
    # model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)
    # # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    # learn_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 1]
    # param_grid = dict(learn_rate=learn_rate)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    # grid_result = grid.fit(trainData, trainLabels)
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    model = Sequential()
    model.add(Dense(11, kernel_initializer="uniform"))
    model.add(Dropout(0.4))
    model.add(Dense(29, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(0.4))
    model.add(Dense(36))
    model.add(Activation("softmax"))

    # train the model using SGD
    print("[INFO] compiling model...")
    sgd = SGD(lr=0.01, momentum=0.6)
    nadam = Nadam(lr=0.001)
    model.compile(loss="categorical_crossentropy", optimizer=nadam, metrics=["accuracy"])

    history = model.fit(trainData, trainLabels, validation_data=(testData, testLabels), epochs=300, batch_size=60, verbose=1)

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=60, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

    # dump the network architecture and weights to file
    print("[INFO] dumping architecture and weights to file...")
    model.save("../sign_language.h5")

if __name__ == "__main__":
    #print(sys.argv[1])
    main(sys.argv[1])  # main takes path to the dataset
