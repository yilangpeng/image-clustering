import os, sys, shutil
import numpy as np
import pandas as pd
import ypoften as of

cwd = os.path.join('/Volumes/Athens/Analysis/Climate/Github',"")
clmethod = "KMeans"

cvmodel = "vgg16 places fc1"
cvmodel = "vgg16 fc1"
cvmodel = "vggface fc6"

savefolder = cvmodel + " " + clmethod + " PCA200"

def save_img(randomid, user, imgname, label, K):
    imgpath1=os.path.join(cwd,'img copy',user, imgname)
    newimgname = str(randomid)+' '+imgname
    imgpath2 = os.path.join(cwd,'img cluster',savefolder, str(K),str(label),newimgname)
    of.copy_file(imgpath1, imgpath2)

for K in range(5, 21):
    print('number of cluster',K)
    filepath = os.path.join(cwd,'img cluster',savefolder,str(K),'label.txt')
    df = pd.read_csv(filepath, sep ='\t', header = None)
    df.columns = ['user','shortcode','posttype','imgname','randomid','label']

    dr = df

    for labelK in range(0, K): 
        dc = dr.loc[dr['label'] == labelK] # select images from each cluster
        dc = dc.iloc[:20,:] # select 20 images from each cluster
        dc.apply(lambda row: save_img(row['randomid'],row['user'],row['imgname'],row['label'], K),axis=1)

    print("DONE"*20)
print("DONE"*20)