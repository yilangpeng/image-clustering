# # -*- coding: utf-8 -*-
# https://gogul09.github.io/software/flower-recognition-deep-learning
# original file name: HA feature extraction-all-VGG16-190510-HZ.py

from pathlib import Path
import numpy as np
import joblib, os, codecs, sys
from keras.preprocessing import image
from keras.applications import vgg16
from keras.models import Model

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.xception import Xception, preprocess_input

# force CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def create_path(filepath):
	filepathfolder = filepath.rsplit('/', 1)[0]
	if not os.path.exists(filepathfolder): os.makedirs(filepathfolder)



# cwd = '/Volumes/Thailand/Research photos/Protest/'
cwd = ''


datadir = "../images/" 


def train_model(model):
	if model == "vgg-4096":
		pretrained_nn = VGG16(weights= "imagenet", include_top=True)
		print (pretrained_nn.summary())
		# pretrained_nn = vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))# Load a pre-trained neural network to use as a feature extractor
		feature_model = Model(input=pretrained_nn.input, output=pretrained_nn.get_layer('fc1').output)
		img_size = (224, 224)
	if model == "resnet50":
		base_model = ResNet50(weights= "imagenet", include_top = True)
		print (base_model.summary())
		feature_model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
		img_size = (224, 224)
	if model == "Xception":
		base_model = Xception(weights= "imagenet", include_top = True)
		print (base_model.summary())
		feature_model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
		img_size = (299, 299)
	# imgnamefile = codecs.open(cwd + 'img filename/' + 'img filename.txt', 'r', encoding = 'utf-8')
	imgnamefile = codecs.open(cwd  + 'img filename.txt', 'r', encoding = 'utf-8')
	# imgnamefile = codecs.open(cwd  + 'img-filename-one-per-single-user.txt', 'r', encoding = 'utf-8')
	for lineindex, line in enumerate(imgnamefile.readlines()[:]):
		#print('-' * 60)
		imgname = line.rstrip('\r\n')
		#print(lineindex, imgname)

		# imgpath = cwd + 'img original/' + imgname 
		imgpath = f"{datadir}{imgname}"
		img = image.load_img(imgpath, target_size=img_size) # Load the image from disk
		image_array = image.img_to_array(img) # Convert the image to a numpy array

		image_expand = np.expand_dims(image_array, 0)
		x_train = vgg16.preprocess_input(image_expand) # normalize image data to 0-to-1 range

		features_x = feature_model.predict(x_train) # extract features for each image

		imgsavepath = cwd + 'img transform/' + model + "/" + imgname + ".dat"
		create_path(imgsavepath)
		joblib.dump(features_x,imgsavepath)
		# print("dump OK=="*20)

# print("DONE"*20)
if __name__ == '__main__':
	train_model("vgg-4096")
	train_model("resnet50")
