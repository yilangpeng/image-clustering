import joblib, codecs, os, sys, random, shutil, time
import numpy as np
import pandas as pd
import ypoften as of

cwd = '/Volumes/Cuba/Analysis/IRA/'
clmethod = "KMeans"

cvmodels = ["vgg16 fc1","vgg16 places fc1","vggface fc6"]

cwd = '/Volumes/Cuba/Analysis/IRA/Github/'

def save_img(randomid, imgname, label, K):
    imgpath1=os.path.join(cwd,'img resize', imgname)
    newimgname = str(randomid) + ' ' + imgname
    imgpath2 = os.path.join(cwd,'img cluster',savefolder, str(K),str(label),newimgname)
    of.copy_file(imgpath1, imgpath2)

for cvmodel in cvmodels:
    features_array = []
    
    savefolder = cvmodel + " " + clmethod + " PCA200"

    for K in [6,8,10]:
        print('number of cluster',K)
        filepath = os.path.join(cwd,'img cluster',savefolder,str(K),'label.txt')
        df = pd.read_csv(filepath, sep ='\t', header = None)
        df.columns = ['imgname','label']
    
        dr = df.sample(frac=1,random_state=42).reset_index(drop=True)
        dr['randomid'] = np.arange(1, dr.shape[0] + 1)
        print(dr)
    
        for labelK in range(0, K): 
            dc = dr.loc[dr['label'] == labelK] # select images from each cluster
            dc = dc.iloc[:20,:] # select 20 images from each cluster
            dc.apply(lambda row: save_img(row['randomid'],row['imgname'],row['label'], K),axis=1)
    
        print("DONE"*20)
    
    print("DONE"*20)