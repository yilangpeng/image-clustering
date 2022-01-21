import joblib, os
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering

cwd = "." # replace with your own folder path
savefolder = '../../fig4/' #subdirectory under cwd to save


def single_file_cbow(cvmodel_path, K = 6):

    basename = os.path.basename(cvmodel_path).split(".")[0]

    sall = basename.split("_")
    save_path = os.path.join(cwd,savefolder, basename + f'_K={int(K)}_label_distance.txt')
    save_path_to_centroids = os.path.join(cwd,savefolder, basename + f'_K={int(K)}_distance.txt')

    print ("save label results to", save_path)

    # K = 6

    X = pd.read_csv(cvmodel_path, sep = "\t", index_col =0)

    print (X)
    cl = KMeans(K) # random_state=0)# n_jobs = -1 )
    cl.fit(X)

    X_dist = cl.transform(X) 
    xd = pd.DataFrame(X_dist)

    # print (cl.cluster_centers_.shape) ## (number of clusters, number of dimensions)
    # print (xd.shape) ## (number of images, distance to clusters) 

    X['labels'] = cl.labels_
    X['filename'] = X.index
    X['distance_to_centroid'] = xd.min(axis=1).tolist()

    X[['filename','labels','distance_to_centroid']].to_csv(save_path, index = False, header = False, sep = "\t")


if __name__ == '__main__':

    for K in [6]:
        single_file_cbow('protest_bbow.csv', K)

