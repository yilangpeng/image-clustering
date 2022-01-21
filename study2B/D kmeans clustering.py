import joblib, os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import ypoften as of

clmethod = "KMeans"

cvmodels = ["vgg16 fc1","vgg16 places fc1","vggface fc6"]

cwd = '/Volumes/Cuba/Analysis/IRA/Github/'

for cvmodel in cvmodels:
    features_array = []

    features_savepath = os.path.join(cwd,'img exfeature','features PCA',cvmodel+'.dat')
    features_array = joblib.load(features_savepath)
    X = pd.DataFrame(features_array)
    print(X)
    
    nd = 200
    X = X.iloc[:,0:nd]
    
    savefolder = cvmodel + ' ' + clmethod + ' PCA' + str(nd)
    
    for K in [6,8,10]:
        print('number of cluster', K)
        cl = KMeans(K, random_state=0)
        cl.fit(X)
        labels = cl.labels_
    
        imgnamefile = os.path.join(cwd,"img filename.txt")
        df = pd.read_csv(imgnamefile, sep ='\t', header=None, names = ["imgname"])
    
        print(df)
    
        df['label'] = labels
        filepath = os.path.join(cwd,'img cluster',savefolder,str(K),'label.txt')
        of.create_path(filepath)
        df.to_csv(filepath, index = None, header = None, sep = '\t') 
    
    print("DONE"*20)