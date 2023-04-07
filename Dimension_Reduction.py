import torch
import time
from torch.optim import Adam
import torch.nn as nn
import sys, os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def img_preprocessing(img):
  img = np.resize(img,(250,250))
  img = img/np.linalg.norm(img)
  return img


def read_data(path,training_classes):

  img_classes = {"shine":0,"sunrise":1,"rainy":2,"foggy":3,"cloudy":4}
  total_size, count = 0, 0

  for c in training_classes:
    total_size+= len(os.listdir(path+"/"+c))
  training_data = np.empty((total_size,250,250),float)
  training_labels = []

  for c in training_classes:
    all_images = os.listdir(path+"/"+c)
    class_labels = [img_classes[c]]*len(all_images)
    training_labels += class_labels
    for idx,val in enumerate(all_images):
      try:
        img_path =  path+"/"+c+"/"+val
        img_data = cv2.imread(img_path)
        training_data[count] = img_preprocessing(img_data)
        count+=1
      except:
        print(img_path)
    
  #print(training_data)

  return (training_data, training_labels, total_size)


def PCA_main(X,total_size):
  X_new = X.reshape(total_size,250*250)
  start = time.time()
  pca = PCA(n_components = 5)
  end = time.time()
  X_final = pca.fit_transform(X_new)
  print("Number of seconds to perform PCA decomposition : ",end-start)
  return X_final


def MDS_main(X,total_size):
  X_new = X.reshape(total_size,250*250)  
  X_new = X_new - X_new.mean()
  start = time.time()
  similarities = euclidean_distances(X_new)
  nmds = manifold.MDS(
    n_components=5,
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
  X_new = X.reshape(total_size,250*250)
  X_new = X_new - X_new.mean()
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


def KNN_classifier(X,y):
    neighbor = [20,30,40]#[10,20,30,40,50,60,70,80,90,100]
    for val in neighbor:
        neigh = KNeighborsClassifier(n_neighbors = val)
        X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=16)
        neigh.fit(X_train,y_train)
        predicted = neigh.predict(X_test)
        print("The accuracy is :",np.sum(predicted==y_test)/len(y_test))


def main():
    training_classes = ["shine","sunrise","rainy","foggy","cloudy"]
    training_data, training_classes, total_size = read_data("data/weather_dataset",training_classes)
   
    #print("KNN with no dimension reduction")
    X, y = shuffle(training_data.reshape(total_size,250*250),training_classes)
    KNN_classifier(X,y)

    """
    print("PCA")
    X_pca = PCA_main(training_data,total_size)
    with open("data/PCA_Data.npy","wb") as f:
                np.save(f,X_pca)
    X_pca = np.load("data/PCA_Data.npy")
    X_pca, y = shuffle(X_pca,training_classes)
    KNN_classifier(X_pca,y)
    """
    
    """
    print("tSNE")
    X_tsne = tsne_main(training_data,total_size)
    with open("data/tSNE_Data.npy","wb") as f:
                np.save(f,X_tsne)
    X_tsne = np.load("data/tSNE_Data.npy")
    X_tsne, y = shuffle(X_tsne,training_classes)
    KNN_classifier(X_tsne,y)
    """

    """
    print("MDS")
    X_mds = MDS_main(training_data,total_size)
    with open("data/MDS_Data.npy","wb") as f:
        np.save(f,X_mds)
    X_mds = np.load("data/MDS_Data.npy")
    X_mds, y = shuffle(X_mds,training_classes)
    KNN_classifier(X_mds,y)
    """

if __name__=="__main__":
    main()
