import os
import cv2 as cv
import numpy as np
from keras.models import load_model
from main import feature_extraction
import pyttsx3
import sys


def textToSpeech(text):
    # One time initialization
    engine = pyttsx3.init()

    # Seting properties before adding things to say
    engine.setProperty('rate', 150)    # Speed percent (can go over 100)
    engine.setProperty('volume', 1.0)  # Volume 0-1

    # Queuing up what to say
    engine.say(text)

    # Flushing the say() queue and playing the audio
    engine.runAndWait()

print("[INFO] Loading model...")
model = load_model("../sign_language.h5")

images = []
labels = []

# loop over our testing images and classify the image
# using our extracted features and saved neural network
def loadTestImage(imagePath):
    test_data = []
    for file_name in os.listdir(imagePath):
        if file_name != ".DS_Store":
            images.append(imagePath + "/" + file_name)

            features = feature_extraction(imagePath + "/" + file_name)
            label = file_name.index("_") + 1
            labels.append(file_name[label])

            if len(features) < 11:
                for i in range(len(features), 11):
                    features.append(0.0)

            if len(features) > 11:
                area = features[-1]
                features = features[0:10]
                features.append(area)

            test_data.append(features)
        else:
            continue

    test_data = np.array(test_data)

    probs = model.predict(test_data)[0]
    prediction = probs.argmax(axis=0)

    # draw the class and probability on the test image and display it
    # to our screen
    label = "{}: {:.2f}%".format(labels[prediction],
                                 probs[prediction] * 100)

    cvimage = cv.imread(images[0])
    cv.putText(cvimage, label, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    cv.imshow("Image", cvimage)
    #textToSpeech(labels[prediction])
    cv.waitKey(0)


def main(testImagePath):
    predictions, image, labels, probs, file_name = loadTestImage(testImagePath)

if __name__ == '__main__':
    testImageFilePath = sys.argv[1]
    loadTestImage(testImageFilePath)
