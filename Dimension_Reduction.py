##https://github.com/coderoda/Cifar-10-Classification-using-scikit-learn/blob/master/smai-mini-project2.py

import pickle, time, sys, os, argparse, ast
import numpy as np

from sklearn.metrics import euclidean_distances
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from sklearn import svm, manifold
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
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
    tst_labels.extend(unpickled["labels"])
    tst_data = np.array(tst_data)
    tst_labels = np.array(tst_labels)

    trn_data = np.array(trn_data,dtype=np.int)
    trn_labels = np.array(trn_labels,dtype=np.int)

    return (trn_data - trn_data.mean(axis=0)), trn_labels, (tst_data - tst_data.mean(axis=0)), tst_labels


def PCA_main(X_new):
  start = time.time()
  pca = PCA(n_components = 5)
  end = time.time()
  X_final = pca.fit_transform(X_new)
  print("Number of seconds to perform PCA decomposition : ",end-start)
  return X_final


def MDS_main(X_new):
  X_new = X_new - X_new.mean()
  start = time.time()
  similarities = euclidean_distances(X_new)
  nmds = manifold.MDS(
    n_components=6,
    metric=False,
    max_iter=1000,
    eps=1e-4,
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


def tsne_main(X):  
  X_new = X - X.mean()
  start = time.time()
  TSNE = manifold.TSNE(
    n_components=2, #1000
    metric="seuclidean",
    method="exact",
    random_state=5,
    n_jobs=2
  )
  modified_X = TSNE.fit_transform(X_new)
  end = time.time()
  print("Time taken for tsne decomposition : ",end-start)
  return modified_X


def KNN_classifier(X,y,dim_reduction):
    neighbor = [1,5,10,20,30,40,50,60,70,80,90,100]
    
    if dim_reduction:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    else:
        X_train = X
        y_train = y
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")

    print("X_train : {}, y_train : {}, X_test : {}, y_test : {}".format(X_train.shape,y_train.shape,X_test.shape,y_test.shape))
    accuracy = []
    for val in neighbor:
        neigh = KNeighborsClassifier(n_neighbors = val)
        neigh.fit(X_train,y_train)
        predicted = neigh.predict(X_test)
        
        #print(predicted.shape,y_test.shape)
        print("Accuracy in KNN : {}, K : {}".format(np.sum(predicted==y_test)/len(y_test),val))
        accuracy.append(np.sum(predicted==y_test)/len(y_test))


def SVM_classifier(X,y,dim_reduction):
    if dim_reduction:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    else:
        X_train = X
        y_train = y
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")

    clf = svm.SVC(decision_function_shape="ovo")
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("The accuracy in SVM classifier is :",np.sum(predicted==y_test)/len(y_test))


def LDA_classifier(X,y,dim_reduction):
    if dim_reduction:
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    else:
        X_train = X
        y_train = y
        X_test = np.load("data/X_test.npy")
        y_test = np.load("data/y_test.npy")

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
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    return X_train,X_test,y_train,y_test


def main(sample_size, classification_model, dimension_reduction):
    
    classification_model = classification_model.split(",")
    dimension_reduction = dimension_reduction.split(",")

    total_size = 50000
    build_cifar = False
    dim_reduction = True

    if build_cifar:
        cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels = load_cifar()
        store_data(cifar_trn_data, cifar_trn_labels, cifar_tst_data, cifar_tst_labels)
    else:
        X_train,X_test,y_train,y_test = load_data()
 
    print("The size of the data being used is : ",sample_size)
    
    choice = input("Classifiy the data directly? [Y/N] ")
    if choice=="Y":
        KNN_classifier(X_train[:sample_size],y_train[:sample_size],False)
        SVM_classifier(X_train[:sample_size],y_train[:sample_size],False) 
        LDA_classifier(X_train[:sample_size],y_train[:sample_size],False) 
    else:
        #print("Running {} dimension reduction algorithms.".format(dimension_reduction))
        for val in dimension_reduction:
            print("Running dimension reduction algorithm : ",val)
            dim_query = val+"_main"
            func1 = globals()[dim_query]
            X = func1(X_train[:sample_size])
            writing_dir = "data/"+dim_query.split("_")[0] + "/X"+"_"+dim_query.split("_")[0]+"_"+str(sample_size)+".npy"
            with open(writing_dir,"wb") as f:
                np.save(f,X)
            
            # Running classificaiton models
            for val in classification_model:
                print("Running classification model :",val)
                model_query = val+"_classifier"
                func2 = globals()[model_query]
                func2(X,y_train[:sample_size],True)
                

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size",type=int,default=500,help="The size of the data you want to use")
    parser.add_argument("--classification",type=str,default='KNN,SVM',help="Enter the models you want to use")
    parser.add_argument("--dim_reduction",type=str,default='PCA,tSNE,NMDS',help="Enter the dimension reduction technique you want to implement")
        
    args=parser.parse_args()
    sample_size = args.size
    classification_model = args.classification
    dimension_reduction = args.dim_reduction
    
    main(sample_size,classification_model,dimension_reduction)

