# coding=utf-8 
"""
Author: Han Zhang
Used to extract bag of visual words (bovw) features
take 40-50 minutes to finish on a 16 core Intel i9 compute, with parallelizaztion

"""
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import MiniBatchKMeans
import glob
import pandas as pd
import pickle
import parmap
from multiprocessing.pool import ThreadPool
import copy

sift = cv2.SIFT_create()


protestdir = "../images/*.jpg"
filelist = glob.glob(protestdir)

print ("number of files", len(filelist))


dictionary_size = 100


desc_list = []
feature_list = []



def read_images(p):
    image = cv2.imread(p)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, dsc = sift.detectAndCompute(gray, None)
    return dsc

feature_list = parmap.map(read_images, filelist, pm_pbar=True, pm_chunksize = 100)

desc_list = [item for sublist in feature_list for item in sublist]

print (len(feature_list))
print (len(desc_list))


print('Creating BoW dictionary using K-Means clustering with k={}..'.format(dictionary_size))
dictionary = MiniBatchKMeans(n_clusters=dictionary_size, max_iter=50, batch_size=20, compute_labels = False)
# dictionary.fit(desc_list)
## partial fit
for k in range(len(filelist) // 500):
    # print (k*500, k*500 + 500)
    dictionary.partial_fit(desc_list[(k*500):(k*500 + 500)])
# return dictionary


# dictionary = build_dictionary(sift, filelist[:10], dictionary_size)
visual_words = dictionary.cluster_centers_
print ("shape of center", visual_words.shape) #  ## (number of clusters, number of dimensions)
# print ("labels:" , len(dictionary.labels_))  # label of each one

def find_index(vector1, vector2):

    distanceList = {}

    for ndx, val in enumerate(vector2):
        distanceD = distance.euclidean(vector1, val)
        distanceList[ndx] = distanceD

    # Then find minimum value and its key.
    index = min(distanceList, key=lambda k: distanceList[k])
    return index

## get histogram

    # hist = []
    # for img in imgs:

lv = len(visual_words)
def image_class(img, center):


    histogram = np.zeros(len(center))
    for each_feature in img:
        ind = find_index(each_feature, center)
        histogram[ind] += 1
        # print (histogram.shape)
        # hist.append(histogram)
    return histogram
    # return hist

filenames_reduces = ["/".join (x.split("/")[-2:] )for x in filelist]
# histogram = image_class(feature_list, visual_words) 
# histogram = parmap.map(image_class, feature_list, pm_pbar=True, pm_chunksize = 20)
# p = ThreadPool(16)
# histogram = p.map(image_class, feature_list)




l = [(feature_list[i], copy.copy(visual_words) ) for i in range(len(feature_list))]
# p = ThreadPool(16)
histogram = parmap.starmap(image_class, l, pm_pbar=True, pm_chunksize = 10)

d = pd.DataFrame(histogram, index = filenames_reduces)
# save the extract features of the protest images into the following file
d.to_csv("protest_bbow.csv", sep = "\t")

