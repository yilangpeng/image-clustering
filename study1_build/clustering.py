import joblib, os
import numpy as np
import pandas as pd
import glob
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
cwd = "" # replace with your own folder path
savefolder = '' #subdirectory under cwd to save


def single_file_cbow(cvmodel_path, K = 6, savefolder = ""):

    basename = os.path.basename(cvmodel_path).split(".")[0]

    sall = basename.split("_")
    save_path = os.path.join(cwd,savefolder, basename + f'_K={int(K)}_label_distance.txt')
    save_path_to_centroids = os.path.join(cwd,savefolder, basename + f'_K={int(K)}_distance.txt')



    print ("save label results to", save_path)


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

def single_file(cvmodel_path, K, savefolder = ""):

    basename = os.path.basename(cvmodel_path).split(".")[0]

    sall = basename.split("_")
    save_path = os.path.join(cwd,savefolder, basename + '_label_distance.txt')
    save_path_to_centroids = os.path.join(cwd,savefolder, basename + '_distance.txt')

    save_path = save_path.replace("=6", f"={K}")
    save_path_to_centroids = save_path_to_centroids.replace("=6", f"={K}")



    # if os.path.exists(save_path):
    #     continue


    try:
        if basename.startswith("pca"):
            dataset = sall[1]

            # # K
            # K = sall[3].split("=")[-1]
            # K = int(K)

            epoch = sall[4].split("=")[-1]

        else:

            dataset = sall[0]


            # K = sall[2].split("=")[-1]
            # K = int(K)

            epoch = sall[3].split("=")[-1]
    except:
        dataset = "politician"
        # K = 6
        epoch = 0


    print ("save label results to", save_path)


    X = joblib.load(cvmodel_path)

    if basename.startswith("pca"):
        X = pd.DataFrame(X)

    else: # pca data are already transposed; otherwise we need to transpose it
        X = pd.DataFrame(X).transpose()


    print (X)
    cl = KMeans(K, random_state=0, n_jobs = -1 )
    cl.fit(X)

    X_dist = cl.transform(X) 
    xd = pd.DataFrame(X_dist)

    # print (cl.cluster_centers_.shape) ## (number of clusters, number of dimensions)
    # print (xd.shape) ## (number of images, distance to clusters) 

    X['labels'] = cl.labels_
    X['filename'] = X.index
    X['distance_to_centroid'] = xd.min(axis=1).tolist()


    X[['filename','labels','distance_to_centroid']].to_csv(save_path, index = False, header = False, sep = "\t")

def single_file_hierarchical(cvmodel_path, K):

    basename = os.path.basename(cvmodel_path).split(".")[0]

    sall = basename.split("_")
    save_path = os.path.join(cwd,savefolder, basename + '_hierarchical.txt')
    # save_path_to_centroids = os.path.join(cwd,savefolder, basename + '_distance.txt')

    save_path = save_path.replace("=6", f"={K}")
    # save_path_to_centroids = save_path_to_centroids.replace("=6", f"={K}")




    try:
        if basename.startswith("pca"):
            dataset = sall[1]

            # # K
            # K = sall[3].split("=")[-1]
            # K = int(K)

            epoch = sall[4].split("=")[-1]

        else:

            dataset = sall[0]


            # K = sall[2].split("=")[-1]
            # K = int(K)

            epoch = sall[3].split("=")[-1]
    except:
        dataset = "politician"
        # K = 6
        epoch = 0


    print ("save label results to", save_path)


    X = joblib.load(cvmodel_path)

    if basename.startswith("pca"):
        X = pd.DataFrame(X)

    else: # pca data are already transposed; otherwise we need to transpose it
        X = pd.DataFrame(X).transpose()


    print (X)
    # cl = AffinityPropagation()
    # cl = SpectralClustering(n_clusters = K, n_jobs = -1, eigen_solver='arpack',
    #                                       affinity="nearest_neighbors")
    cl = AgglomerativeClustering(n_clusters = K)
    l = cl.fit_predict(X)
    print (l)
    print (np.bincount(l))
    # X_dist = cl.fit_predict(X) 
    # xd = pd.DataFrame(X_dist)

    # print (cl.cluster_centers_.shape) ## (number of clusters, number of dimensions)
    # print (xd.shape) ## (number of images, distance to clusters) 

    X['labels'] = l
    X['filename'] = X.index
    # X['distance_to_centroid'] = xd.min(axis=1).tolist()


    X[['filename','labels']].to_csv(save_path, index = False, header = False, sep = "\t")
    # X[['filename','labels','distance_to_centroid']].to_csv(save_path, index = False, header = False, sep = "\t")


def single_file_gaussian_mixture(cvmodel_path, K, savefolder = ""):

    basename = os.path.basename(cvmodel_path).split(".")[0]

    sall = basename.split("_")
    save_path = os.path.join(cwd,savefolder, basename + '_label_gaussianmixture.txt')
    save_path_to_centroids = os.path.join(cwd,savefolder, basename + '_label_morenthan1category.txt')



    try:
        if basename.startswith("pca"):
            dataset = sall[1]

            # K
            K = sall[3].split("=")[-1]
            K = int(K)

            epoch = sall[4].split("=")[-1]

        else:

            dataset = sall[0]


            K = sall[2].split("=")[-1]
            K = int(K)

            epoch = sall[3].split("=")[-1]
    except:
        dataset = "politician"
        K = 6
        epoch = 0


    print ("save label results to", save_path)


    X = joblib.load(cvmodel_path)

    if basename.startswith("pca"):
        X = pd.DataFrame(X)

    else: # pca data are already transposed; otherwise we need to transpose it
        X = pd.DataFrame(X).transpose()


    # cl = KMeans(K, random_state=0, n_jobs = -1 )
    # cl.fit(X)

    # X_dist = cl.transform(X) 
    # xd = pd.DataFrame(X_dist)


    cl = mixture.GaussianMixture(n_components = K,
                                      covariance_type="full")

    # cl = mixture.BayesianGaussianMixture(n_components = K,
    #                                   covariance_type="full",
    #                                   weight_concentration_prior = 0.01)
    cl.fit(X)

    labels = cl.predict(X)


    yb = cl.predict_proba(X)
    print (np.sum(yb, axis=0))

    ## cluster means
    # print (cl.means_)


    dl = []
    multi_cat = []
    for i,xx in enumerate(labels):
        # print (i, xx)
        dist = dist = np.linalg.norm(cl.means_[xx,:] - X.iloc[i,:])
        # print yb[i,:]
        if (yb[i,:] > 0.2).sum()>1:
            multi_cat.append(i)
        dl.append(dist)


    X1 = pd.DataFrame()
    X1['filename'] = X.index
    X1['labels'] = labels
    X1['distance_to_centroid'] = dl

    print (multi_cat)

    df = pd.concat([X1, pd.DataFrame(yb)], axis=1)
    df.to_csv(save_path, index = False, header = False, sep = "\t")


    # # files that tend to belong to multiple category
    df.iloc[multi_cat].to_csv(save_path_to_centroids, index = False, header = False, sep = "\t")


if __name__ == '__main__':


    K = 6

    # Fig 4:
    # cvmodel_path = "bovw/protest_bbow.csv"
    # single_file_cbow(cvmodel_path, K, "../fig4/")


    # # Fig 5
    # cvmodel_path = "self-supervised/protest_ARCH=vgg16_K=6_EPOCH=5.pickle"
    # single_file(cvmodel_path, K, savefolder = "../fig5/")

    # Fig 6
    # cvmodel_path = "self-supervised/protest_ARCH=vgg16_K=6_EPOCH=5.pickle"
    # single_file(cvmodel_path, K, savefolder = "../fig6/")


    # # Fig 7
    # cvmodel_path = "self-supervised/protest_ARCH=vgg16_K=6_EPOCH=425.pickle"
    # single_file(cvmodel_path, K, savefolder = "../fig7/")


    # for K in [6, 8, 10]:
        # single_file_cbow('/home/han/Dropbox/Collaborations/with_Yilang_Peng/YP protest/img_features_bovw/protest_bbow.csv', K)

    # K = 6
    # single_file_hierarchical(cvmodel_path, K)


    # Fig B2
    cvmodel_path = "self-supervised/protest_ARCH=vgg16_K=6_EPOCH=425.pickle"

    single_file_gaussian_mixture(cvmodel_path, K, savefolder = "../figB2")