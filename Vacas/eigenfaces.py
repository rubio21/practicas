import cv2
import argparse
import os
import imutils
import copy
import numpy as np

carpeta = 'images/'
size_images = (250, 250)
def train_model():
    peopleList = os.listdir(carpeta)
    labels = []
    facesData = []
    label = 0
    for nameDir in peopleList:
        personPath = carpeta + nameDir
        for fileName in os.listdir(personPath):
            labels.append(label)
            img=cv2.imread(personPath + '/' + fileName,0)
            resized = cv2.resize(img, size_images, interpolation=cv2.INTER_AREA)
            facesData.append(resized)
        label = label + 1
    # face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer = cv2.face.EigenFaceRecognizer_create()
    face_recognizer.train(facesData, np.array(labels))
    face_recognizer.write('LBPH_model.xml')
    # face_recognizer.write('eigen_model.xml')



train_model()
imagePaths = os.listdir(carpeta)
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.EigenFaceRecognizer_create()
face_recognizer.read('LBPH_model.xml')
# face_recognizer.read('eigen_model.xml')
for i in os.listdir('FUERA'):
    img = cv2.imread('FUERA/' + i)
    resized = cv2.resize(img, size_images)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    result = face_recognizer.predict(gray)
    print("PHOTO: ", i, "   PREDICTION: ", imagePaths[result[0]], " CONFIDENCE: ", result[1])