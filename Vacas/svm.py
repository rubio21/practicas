import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics, gaussian_process, tree, neural_network, svm, neighbors
import shutil
from sklearn.linear_model import SGDClassifier

carpeta = os.listdir('dataset/malManual')
d=[]
for foto in carpeta:
    img = cv2.cvtColor(cv2.imread('dataset/malManual/' + foto), cv2.COLOR_BGR2GRAY)
    hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img= list(np.concatenate(hist_img, axis=0))
    d.append([hist_img, 0, foto])

carpeta = os.listdir('dataset/bienManual')
for foto in carpeta:
    img = cv2.imread('dataset/bienManual/' + foto)
    img = cv2.cvtColor(cv2.imread('dataset/bienManual/' + foto), cv2.COLOR_BGR2GRAY)
    hist_img = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist_img= list(np.concatenate(hist_img, axis=0))
    d.append([hist_img, 1, foto])

X = np.stack(np.array(d, dtype="object")[:,0], axis=0)
Y = np.stack(np.array(d, dtype="object")[:,1], axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
# clf = svm.SVC(kernel="rbf", class_weight='balanced')
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

index_pred = 0
acierto_index=[0,0]
error_index=[0,0]
for foto in X_test:
    for i in range(len(X)):
       if (X[i] == foto).all():
           if y_pred[index_pred]==0:
               if y_pred[index_pred] == d[i][1]:
                   shutil.copyfile('homografias/' + d[i][2], 'svm_mal/acierto_' + str(acierto_index[0]) + '.png')
                   acierto_index[0]+=1
               else:
                   shutil.copyfile('homografias/' + d[i][2], 'svm_mal/error_' + str(error_index[0]) + '.png')
                   error_index[0]+=1
           else:
               if y_pred[index_pred] == d[i][1]:
                   shutil.copyfile('homografias/' + d[i][2], 'svm_bien/acierto_' + str(acierto_index[1]) + '.png')
                   acierto_index[1]+=1
               else:
                   shutil.copyfile('homografias/' + d[i][2], 'svm_bien/error_' + str(error_index[1]) + '.png')
                   error_index[1] += 1
           index_pred+=1