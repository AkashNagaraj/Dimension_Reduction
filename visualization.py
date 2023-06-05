import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

def KNN_variations():
    pca_data = [25,23.5,24.5,27,28,27,27.5,27,26.5,25,25.5,25]
    nmds_data = [5,9.5,9,11.5,10,9,9.5,9.5,8.5,11,10,8.5]
    K_values = [1,5,10,20,30,40,50,60,70,80,90,100]

    df = pd.DataFrame({'PCA': pca_data,'NMDS': nmds_data,'K': K_values})
    fig = px.line(df,x="K",y=df.columns[0:2],markers=True,title="Accuracy score for different nearest neighbors")
    fig.update_layout(
        font_family="Courier New",
        title_x=0.5,
        title_font_family="Times New Roman",
        title_font_color="rgb(10,29,81)",
        yaxis_title="Accuracy %",
        xaxis_title="# Neighbors",
    )

    fig.write_image("../Dimension_Reduction_IEEE/PCA vs NMDS.png")

def accuracy():
    KNN = [22.6,28.36,29.55,31.34,32.79]#,29.86]
    SVM = [34.38,43.86,47.14,50.39,51.68]
    LDA = [18.5,18.98,26.03,32.05,34.64]
    image_size = [1000,5000,10000,20000,30000]#,50000]

    data = pd.DataFrame({"KNN":KNN,"SVM":SVM,"LDA":LDA,"IMAGE":image_size})
    sns.set_style("darkgrid")
    sns.lineplot(data=data,x="IMAGE",y="KNN",label="KNN").set(title="Accuracy Scores",xlabel="Number of training images",ylabel="Accuracy %")
    sns.lineplot(data=data,x="IMAGE",y="SVM",label="SVM")
    sns.lineplot(data=data,x="IMAGE",y="LDA",label="LDA")
    plt.savefig("../Dimension_Reduction_IEEE/Accuracy Scores.png")

    plt.show()


# https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels
def tsne_plot():
    X = np.load("data/tsne/X_tsne_5000.npy")
    label = np.load("data/y_train.npy")[:5000]
    colors= ["navy","darkblue","slateblue","rebeccapurple","indigo","darkmagenta","crimson","lightseagreen","yellow","darkorange"]
    plt.style.use('dark_background')    
    fig = plt.figure(figsize=(10,10))
    plt.scatter(X[:,0],X[:,1],c=label,cmap=matplotlib.colors.ListedColormap(colors))
    plt.title("Resulting t-SNE projection")

    cb = plt.colorbar()
    loc = np.arange(0,max(label),max(label)/float(len(colors)))
    cb.set_ticks(loc)
    #cb.set_ticklabels(colors) 
    plt.savefig("../Dimension_Reduction_IEEE/tsne_projection.png")
    plt.show()
    
    

tsne_plot()
#accuracy()
#KNN_variations()
