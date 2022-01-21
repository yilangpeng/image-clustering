import os, sys, joblib
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model
import numpy as np
import ypoften as of

cvmodel = "vgg16 fc1"

if cvmodel == "vgg16 fc1":
    from keras.applications.vgg16 import VGG16, preprocess_input
    img_size = (224, 224)
    base_model = VGG16(weights= "imagenet", include_top=True)
    feature_model = Model(base_model.input, base_model.get_layer('fc1').output)
    print(base_model.summary())

cwd = '/Volumes/Cuba/Analysis/IRA/Github/'

imgfile = os.path.join(cwd, "img filename.txt")
imgfile = open(imgfile, 'r')
imglines = imgfile.readlines()[:]

for i, imgline in enumerate(imglines[:]):
    imgname = imgline.rstrip('\r\n')
    print(i, imgname, '*'*20)

    imgpath = os.path.join(cwd,"img resize", imgname)
    exfolder = os.path.join(cwd,"img exfeature",cvmodel,"")
    featuresavepath = os.path.join(exfolder, imgname + ".dat")
    of.create_path(featuresavepath)
    print(featuresavepath)

    img = image.load_img(imgpath, target_size=img_size) # Load the image from disk
    image_array = image.img_to_array(img) # Convert the image to a numpy array

    image_expand = np.expand_dims(image_array, 0)
    x_train = preprocess_input(image_expand) # normalize image data to 0-to-1 range

    features_x = feature_model.predict(x_train) # extract features for each image

    joblib.dump(features_x[0], featuresavepath)
    print("dump OK=="*20)

print("DONE"*20)
