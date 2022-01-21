import joblib, codecs, os, sys, random, shutil
import numpy as np
import ypoften as of
import pandas as pd

cvmodels = ["vgg16 fc1","vgg16 places fc1","vggface fc6"]

cwd = '/Volumes/Cuba/Analysis/IRA/Github/'

for cvmodel in cvmodels:
    features_array = []

    imgfile = os.path.join(cwd, "img filename.txt")
    imgfile = open(imgfile, 'r')
    imglines = imgfile.readlines()[:]
    
    for i, imgline in enumerate(imglines[:]):
        imgname = imgline.rstrip('\r\n')
        print(i, imgname, '*'*20)
    
        ex_feature_path = os.path.join(cwd, 'img exfeature', cvmodel, imgname + ".dat")
        imgexfeatures = joblib.load(ex_feature_path)
        print(imgexfeatures.shape)
    
        flatten_features = np.ndarray.flatten(imgexfeatures)
        print(flatten_features.shape)
        features_array.append(flatten_features)
    
    features_array = np.array(features_array)
    print(features_array.shape)
    features_savepath = os.path.join(cwd, 'img exfeature','features combine', cvmodel + ".dat")
    of.create_path(features_savepath)
    joblib.dump(features_array,features_savepath); print("dump OK=="*20)

print("DONE"*20)