import os, codecs, joblib, sys
from keras.preprocessing import image
from keras.models import Model
from keras.applications.vgg16 import preprocess_input

import numpy as np
import ypoften as of

# select the pretrained model
cvmodel = "vgg16 places fc1"

from vgg16_places_365 import VGG16_Places365
img_size = (224, 224)
base_model = VGG16_Places365(weights='places', include_top=True, input_shape=(224, 224, 3))
feature_model = Model(base_model.input, base_model.get_layer('fc1').output)
print(feature_model.summary())

cwd = os.path.join('/Volumes/Athens/Analysis/Climate/Github',"")
# please change this to the folder that has all the images

imgfile = os.path.join(cwd, 'img filename study2A.txt')
imgfile = open(imgfile, 'r', encoding="utf-8")

for j, line in enumerate(imgfile.readlines()[1:]):
    user, shortcode, posttype, imgname = line.rstrip('\r\n').split('\t')[:4]
    print(j, imgname)

    imgpath = os.path.join(cwd, "img copy", user, imgname)
    img = image.load_img(imgpath, target_size=img_size) # load the image
    image_array = image.img_to_array(img) # convert the image to a numpy array

    image_expand = np.expand_dims(image_array, 0)
    x_train = preprocess_input(image_expand) # normalize image data to 0-1 range

    features_x = feature_model.predict(x_train) # extract features

    featuresavepath = os.path.join(cwd,"img exfeature",cvmodel,user,imgname + ".dat")
    exfolder = os.path.join(cwd,"img exfeature",cvmodel, user, "")
    featuresavepath = os.path.join(exfolder, imgname + ".dat") # save features

    of.create_path(featuresavepath)
    joblib.dump(features_x[0], featuresavepath)

print("DONE"*20)
