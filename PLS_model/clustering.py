from sklearn.decomposition import PCA
import phate
import seaborn as sns
import umap
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_TSNE(X_features, y_label, ax, n_components=2, verbose=1, random_state=123):
    tsne = TSNE(n_components=n_components, verbose=verbose, random_state=random_state)
    z = tsne.fit_transform(X_features)
    df = pd.DataFrame()
    df["y"] = y_label
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    scatter_plot = sns.scatterplot(x="comp-1", 
                                y="comp-2", 
                                hue=df.y.tolist(),
                                palette=sns.color_palette("hls", len(set(y_label))),
                                data=df, 
                                ax=ax
                                )

    
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.show()

def plot_PCA(X_features, y_label, ax):
    #convert the features into the 2 top features
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_features)
    #principalDf = pd.DataFrame(data = principalComponents, columns =['principal component 1', 'principal component 2'])
    #principalDf.head(5)
    #data[['group']].head()
    #finalDf = pd.concat([principalDf, data[['group']]], axis = 1)
    df = pd.DataFrame()
    df["y"] = y_label
    df["comp-1"] = principalComponents[:,0]
    df["comp-2"] = principalComponents[:,1]

    scatter_plot = sns.scatterplot(x="comp-1", 
                                y="comp-2", hue=df.y.tolist(),
                                palette=sns.color_palette("hls", len(set(y_label))),
                                data=df, 
                                ax=ax
                                )

def plot_PHATE(X_features, y_label, ax, knn=5, decay=15, t=12):
    phate_op = phate.PHATE()
    phate_op.set_params(knn=knn, decay=decay, t=t)
    #knn : Number of nearest neighbors (default: 5).
    #Increase this (e.g. to 20) if your PHATE embedding appears verydisconnected.
    #You should also consider increasing knn if your dataset is extremely large(e.g. >100k cells)
    #decay : Alpha decay (default: 15). Decreasing decay increases connectivity on the graph, increasing decay decreases connectivity.
    #This rarely needs to be tuned. Set it to None for a k-nearest neighbors kernel.

    data_phate = phate_op.fit_transform(X_features)
    result_phate = pd.DataFrame()
    result_phate["y"] = y_label
    result_phate["PHATE-1"] = data_phate[:,0]
    result_phate["PHATE-2"] = data_phate[:,1]

    sns.scatterplot(x="PHATE-1", 
                    y="PHATE-2", 
                    hue=result_phate.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y_label))),
                    data=result_phate,
                    ax=ax
                    )
    
def plot_UMAP(X_features, y_label, ax):
    plt.rcParams['figure.figsize'] = (6,5)
    #mapper = umap.UMAP().fit(X_features)
    #umap.plot.points(mapper, labels=y_label)
    embedding = umap.UMAP(n_neighbors=10,
    min_dist=0.5,n_components=2,
    metric='correlation').fit_transform(X_features)
    result = pd.DataFrame()
    result["y"] = y_label
    result["UMAP-1"] = embedding[:,0]
    result["UMAP-2"] = embedding[:,1]

    sns.scatterplot(x="UMAP-1", 
                    y="UMAP-2", 
                    hue=result.y.tolist(),
                    palette=sns.color_palette("hls", len(set(y_label))),
                    data=result, 
                    ax=ax
                    )

