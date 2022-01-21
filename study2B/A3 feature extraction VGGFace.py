import os, joblib
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model

import ypoften as of

cvmodel = 'vggface fc6'

from keras_vggface.vggface import VGGFace
vggface = VGGFace(model='vgg16') # or VGGFace() as default
feature_model = Model(vggface.input, vggface.get_layer('fc6').output)
img_size = (224, 224)
print(feature_model.summary())

cwd = '/Volumes//Cuba/Analysis/IRA/Github/'

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

