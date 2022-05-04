import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics, gaussian_process, tree, neural_network, svm

def sobel_h(img, threshold):
    G_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    rows = np.size(img, 0)
    columns = np.size(img, 1)
    mag = np.zeros(img.shape)
    for i in range(0, rows - 2):
        for j in range(0, columns - 2):
            h = sum(sum(G_y * img[i:i + 3, j:j + 3]))  # horizon
            mag[i + 1, j + 1] = h
    for p in range(0, rows):
        for q in range(0, columns):
            if mag[p, q] < threshold:
                mag[p, q] = 0
    return mag


def train_model():
    carpeta = os.listdir('CantonadaPiezaCuadrada01')
    d=[]
    for foto in carpeta:
        img = cv2.resize(cv2.imread('CantonadaPiezaCuadrada01/' + foto, 0),(648,486))
        mag_h = sobel_h(img, 70)
        opening = cv2.morphologyEx(mag_h, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=2)
        x = np.sum(opening, axis=1)
        d.append([x, 0, foto])
        print('CantonadaPiezaCuadrada01/' + foto)

    print('-------------------------')
    carpeta = os.listdir('CantonadaPiezaCuadrada02')
    for foto in carpeta:
        img = cv2.resize(cv2.imread('CantonadaPiezaCuadrada02/' + foto, 0),(648,486))
        mag_h = sobel_h(img, 70)
        opening = cv2.morphologyEx(mag_h, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=2)
        x = np.sum(opening, axis=1)
        d.append([x, 1, foto])
        print('CantonadaPiezaCuadrada02/' + foto)


    X = np.stack(np.array(d, dtype="object")[:,0], axis=0)
    Y = np.stack(np.array(d, dtype="object")[:,1], axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    clf = svm.SVC(kernel="rbf")
    # clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    filename = 'finalized_model.sav'
    pickle.dump(clf, open(filename, 'wb'))


# train_model()
filename = 'svm.sav'
clf = pickle.load(open(filename, 'rb'))
carpeta = os.listdir('CantonadaPiezaCuadrada01')
for foto in carpeta[125:150]:
    img = cv2.resize(cv2.imread('CantonadaPiezaCuadrada01/' + foto, 0),(648,486))
    cv2.imshow('img',img)
    mag_h = sobel_h(img, 70)
    cv2.imshow('mag_h',mag_h)
    opening = cv2.morphologyEx(mag_h, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=2)
    cv2.imshow('opening',opening)
    x = np.sum(opening, axis=1)
    j = clf.predict([x])
    print('CantonadaPiezaCuadrada01/' + foto + '    Prediction: ' + str(j[0]))
    cv2.waitKey()

