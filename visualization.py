import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def KNN_variations():
    pca_data = [19.4,22.5,24,26.7,26.3,27.4,27.2,26,27.1,26.1,26.5,26.2]
    nmds_data = [9.6,10.7,9.7,11.1,10.5,9.9,9.8,9.5,9.3,9,10,9.7]
    K_values = [1,5,10,20,30,40,50,60,70,80,90,100]
    
    data = pd.DataFrame({'PCA': pca_data,'NMDS': nmds_data,'K': K_values})
    sns.set_style("darkgrid")
    sns.lineplot(data=data,x="K",y="PCA",label="PCA",marker="o").set(title="Accuracy score for different K values",xlabel="K values",ylabel="Accuracy %")
    sns.lineplot(data=data,x="K",y="NMDS",label="NMDS",marker="o")

    plt.savefig("../Dimension_Reduction_IEEE/PCA vs NMDS.png")
    plt.show()

def accuracy():
    KNN = [22.68,23.8,28.9,30.5,31.47,32.79]
    SVM = [33.63,35.64,43.84,47.48,50.45,51.68]
    LDA = [19.76,18.5,18.88,26.06,32.05,34.65]
    CNN = [27.79,41.08,40.87,40.86,41.64,40] # last value incorrect
    image_size = [500,1000,5000,10000,20000,30000]

    data = pd.DataFrame({"KNN":KNN,"CNN":CNN,"SVM":SVM,"LDA":LDA,"IMAGE":image_size})
    sns.set_style("darkgrid")
    sns.lineplot(data=data,x="IMAGE",y="KNN",label="KNN",marker="o").set(title="Accuracy Scores",xlabel="Number of training images",ylabel="Accuracy %")
    sns.lineplot(data=data,x="IMAGE",y="SVM",label="SVM",marker="o")
    sns.lineplot(data=data,x="IMAGE",y="CNN",label="CNN",marker="o")
    sns.lineplot(data=data,x="IMAGE",y="LDA",label="LDA",marker="o")
    plt.savefig("../Dimension_Reduction_IEEE/Accuracy Scores.png")
    plt.show()


def tsne_plot():
    X = np.load("data/tSNE/X_tSNE_5000.npy")
    label = np.load("data/y_train.npy")[:5000]
    colors= ["deeppink","darkblue","slateblue","rebeccapurple","indigo","darkmagenta","crimson","lightseagreen","yellow","orangered"]

    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
    plt.title("Resulting 2D t-SNE projection")

    cb = plt.colorbar()
    plt.savefig("../Dimension_Reduction_IEEE/tsne_projection.png")
    plt.show()
    
    

tsne_plot()
accuracy()
KNN_variations()
