import os, joblib
import numpy as np
import ypoften as of

cvmodel = "vgg16 fc1"
cvmodel = "vgg16 places fc1"
cvmodel = "vggface fc6"

cwd = os.path.join('/Volumes/Athens/Analysis/Climate/Github',"")

features_array = []

imgfile = os.path.join(cwd, 'img filename study2A.txt')
imgfile = open(imgfile, 'r', encoding="utf-8")

for j, line in enumerate(imgfile.readlines()[1:10]):

    user, shortcode, posttype, imgname = line.rstrip('\r\n').split('\t')[:4]

    print(j, user, imgname)

    ex_feature_path = os.path.join(cwd, 'img exfeature', cvmodel, user, imgname + ".dat")

    imgexfeatures = joblib.load(ex_feature_path)
    flatten_features = np.ndarray.flatten(imgexfeatures)
    features_array.append(flatten_features)

features_array = np.array(features_array)
features_savepath = os.path.join(cwd, 'img exfeature','features combine', cvmodel + ".dat")
of.create_path(features_savepath)
joblib.dump(features_array,features_savepath)
print("dump OK=="*20)
