##https://github.com/coderoda/Cifar-10-Classification-using-scikit-learn/blob/master/smai-mini-project2.py

import pickle, time, sys, os
import numpy as np

from sklearn import svm
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_cifar():
    trn_data, trn_labels, tst_data, tst_labels = [], [], [], []

    def unpickle(file):
        with open(file, 'rb') as fo:
            data = pickle.load(fo, encoding='latin1')
        return data

    for i in range(5):
        batchName = './data/cifar_10/data_batch_{0}'.format(i + 1)
        unpickled = unpickle(batchName)
        trn_data.extend(unpickled['data'])
        trn_labels.extend(unpickled['labels'])
    
    unpickled = unpickle('./data/cifar_10/test_batch')
    tst_data.extend(unpickled["data"])
    tst_labels.extend(unpickled["data"])
    tst_data = np.array(tst_data)
    tst_labels = np.array(tst_labels)

    trn_data = np.array(trn_data)
    trn_labels = np.array(trn_labels)
    return (trn_data - trn_data.mean(axis=0)), trn_labels, (tst_data - tst_data.mean(axis=0)), tst_labels


def PCA_main(X_new,total_size):
  start = time.time()
  pca = PCA(n_components = 32)
  end = time.time()
  X_final = pca.fit_transform(X_new)
  print("Number of seconds to perform PCA decomposition : ",end-start)
  return X_final


def MDS_main(X_new,total_size):
  X_new = X_new - X_new.mean()
  start = time.time()
  similarities = euclidean_distances(X_new)
  nmds = manifold.MDS(
    n_components=6,
    metric=False,
    max_iter=1000,
    eps=1e-8,
    dissimilarity="precomputed",
    random_state=5,
    n_jobs=1,
    n_init=1,
    normalized_stress="auto",
  )
  X_modified = nmds.fit_transform(similarities)
  end = time.time()
  print("Time taken for MDS decomposition : ",end-start)
  # Rescale the data
  X_modified *= np.sqrt((X_new**2).sum()) / np.sqrt((X_modified**2).sum())
  return X_modified


def tsne_main(X,total_size):
  X_new = X - X.mean()
  start = time.time()
  TSNE = manifold.TSNE(
    n_components=1, #1000
    metric="seuclidean",
    method="exact",
    random_state=5,
    n_jobs=1
  )
  modified_X = TSNE.fit_transform(X_new)
  end = time.time()
  print("Time taken for tsne decomposition : ",end-start)
  return modified_X


def KNN_classifier(X_train,y_train,X_test,y_test):
    neighbor = [20,30,40]#[10,20,30,40,50,60,70,80,90,100]
    for val in neighbor:
        neigh = KNeighborsClassifier(n_neighbors = val)
        neigh.fit(X_train,y_train)
        predicted = neigh.predict(X_test)
        
        print(predicted.shape,y_test.shape)
        print("The accuracy in KNN is :",np.sum(predicted==y_test)/len(y_test))


def SVM_classifier(X_train,y_train,X_test,y_test):
    clf = svm.SVC(decision_function_shape="ovo")
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("The accuracy in SVM classifier is :",np.sum(predicted==y_test)/len(y_test))


def LDA_classifier(X_train,y_train,X_test,y_test):
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("The accuracy in LDA classifier is :",np.sum(predicted==y_test)/len(y_test))


def store_data(cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels):
    
    global X_train,y_train,X_test,y_test
    
    X_train, y_train = cifar_trn_data, cifar_trn_labels
    X_train, y_train = shuffle(X_train,y_train)
    with open("data/X_train.npy","wb") as f:
        np.save(f,X_train)
    with open("data/y_train.npy","wb") as f:
        np.save(f,y_train)
    print("The shape of cifar training is : {} and : {}".format(cifar_trn_data.shape,cifar_trn_labels.shape)) # (50000, 3072)
    
    X_test, y_test = cifar_tst_data, cifar_tst_labels
    X_test, y_test = shuffle(X_test,y_test)
    with open("data/X_test.npy","wb") as f:
        np.save(f,X_test)
    with open("data/y_test.npy","wb") as f:
        np.save(f,y_test)
    print("The shape of cifar testing is : {} and : {}".format(cifar_tst_data.shape,cifar_tst_labels.shape))


def load_data():
    global X_train,y_train,X_test,y_test
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")


def main():
    
    total_size, sample_size = 50000,10000
    
    build_cifar = True

    if build_cifar:
        cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels = load_cifar()
        store_data(cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels)
    else:
        load_data()
    
    KNN_classifier(X_train[:sample_size],y_train[:sample_size],X_test,y_test)
    SVM_classifier(X_train[:sample_size],y_train[:sample_size],X_test,y_test) 
    LDA_classifier(X_train[:sample_size],y_train[:sample_size],X_test,y_test)  

    sys.exit()

    print("Running PCA")
    X_pca = PCA_main(X,total_size)
    KNN_classifier(X_pca,y)
    print("Completed PCA")
    
    print("Runnning tSNE")
    X_tsne = tsne_main(X,total_size)
    KNN_classifier(X_tsne,y)
    print("Completed tSNE")
    
    print("MDS")
    X_MDS = MDS_main(X,total_size)
    KNN_classifier(X_MDS,y)
    print("Completed MDS")
    

if __name__=="__main__":
    main()
