import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plots():
    pca_data = [62.133, 61.0667, 60, 60.533, 60.8, 59.466, 57.333, 57.33, 56.533, 55.466]
    nmds_data = [25.066,22.933,24.5353,24.5353,24.8,24,21.866,24,25.066,22.666]
    K_values = [10,20,30,40,50,60,70,80,90,100]

    df = pd.DataFrame({'PCA': pca_data,'NMDS': nmds_data,'K': K_values})

    sns.lineplot(data=df,x="K",y="PCA")
    plt.show()

plots()
