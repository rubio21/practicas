import cv2
import numpy as np
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import svm, tree, neural_network, neighbors, gaussian_process
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

def classifier_selection(classifier_name):
    
    if(classifier_name == 'SVM'):
        classifier = svm.SVC(kernel="rbf", class_weight='balanced')
    elif(classifier_name == 'KNN'):
        classifier = neighbors.KNeighborsClassifier(n_neighbors=2)
    elif(classifier_name == 'MLP'):
        classifier = neural_network.MLPClassifier()
    elif(classifier_name == 'DT'):
        classifier = tree.DecisionTreeClassifier()
    elif(classifier_name == 'GPC'):
        classifier = gaussian_process.GaussianProcessClassifier()
    elif(classifier_name == 'SGD'):
        classifier = SGDClassifier(penalty='elasticnet', class_weight='balanced')
    return classifier

if __name__ == '__main__':
    # load data and gt
    class_method = 'SVM'
    method = "triangle"
    gt_df = pd.read_csv("./dataset/gt_3.csv")    
    data_df = pd.read_csv("./dataset/" + method + "-20-1_2.csv")
    do_pca = False
    plot_decision = False
    
    data_array = data_df.values
    gt_array = gt_df.values.flatten()
    
    #remove first column of data array as it contains the name string
    data_array = data_array[:,1:].astype(np.float)
    
    # split train test using sklearn
    train_x, test_x, train_y, test_y = train_test_split(data_array, gt_array, stratify=gt_array, test_size=0.3, random_state=42)
    
    #preprocess data
    scaler = preprocessing.StandardScaler().fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)
    data_array = scaler.transform(data_array)
    
    if(do_pca):
        # generate pca
        pca_trans = PCA(n_components=2).fit(train_x)
        print(pca_trans.score(train_x))
        train_x = pca_trans.transform(train_x)
        test_x = pca_trans.transform(test_x)
        data_array = pca_trans.transform(data_array)
    
    # create svm for classification
    classifier = classifier_selection(classifier_name=class_method)
    classifier.fit(train_x, train_y)
    
    if(plot_decision):
        plot_decision_regions(X=data_array, y=gt_array, clf=classifier, legend=2)
        plt.show()

    
    print("total accuracy")
    print("train: ", classifier.score(train_x, train_y))
    print("test", classifier.score(test_x, test_y))
    
    test_pred = classifier.predict(X=test_x)
    print(classification_report(test_y, test_pred))
    
    # save all predictions
    all_pred = classifier.predict(X=data_array)
    print(len(all_pred))
    pd.DataFrame(all_pred).to_csv("./dataset/predictions_" + method + "_" + class_method + ".csv", index=False, header=True)