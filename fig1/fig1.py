
import codecs, os, glob
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import numpy as np
from random import sample 
import random
from collections import defaultdict
import glob
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def fill_square(im, tbw):
    size = (tbw, tbw)
    bg = Image.new('RGB', size, "white")  # create a background image  
    im.thumbnail(size, Image.ANTIALIAS)
    w, h = im.size 
    bg.paste(im, (int((size[0]-w)/2), int((size[1]-h)/2)))  
    return(bg)


w_sq = 300 # width/height of image
w_clusterid = 60 # width of text for cluster id
h_imgid = 60 # height of gap
fntcluster = ImageFont.truetype('/Library/Fonts/Arial.ttf', 60)


# clustering method
cluster_method = "KMeans" 



# where image folder is located
cwd = "../study1_build"
savefolder = "."


def random_visual_file(file):
    basename = os.path.basename(file)

    sall = basename.split("_")
    pca = ""
    K = 6
    ncol = 10 # how many images to show in each row

    nrow = 1 # number of rows within each category
    nimg = ncol * nrow # total number of image in each cluster to show
    large_w = w_sq*ncol + w_clusterid # total width of the entire image
    large_h = (w_sq + h_imgid) * nrow * K # total height of the entire image
    large = Image.new('RGB', (large_w, large_h), "white")

    large_savepath = os.path.join(savefolder,  "purelyrandom.png")
    print (large_savepath)


    # * read labeling ressults
    results = open(file, 'r').readlines()
    imgfiles = [x.split()[0] for x in results]
    imgfiles = [os.path.join(cwd,  x) for x in imgfiles]


    for label in range(0, K): 

        for j in range(ncol): 

            imgfile = random.choice(imgfiles)

            d = ImageDraw.Draw(large)

            # for j, imgfile in enumerate(imgfiles):
            jw = j%ncol; jh = j//ncol
            img = Image.open(imgfile)
            im_sq = fill_square(img, w_sq)
            im_sq_x = jw*w_sq + w_clusterid
            im_sq_y = h_imgid + jh* (w_sq + h_imgid) + label * nrow * (w_sq + h_imgid)
            large.paste(im_sq, (im_sq_x, im_sq_y))

    large.save(large_savepath)


if __name__ == '__main__':
 

    file = '../study1_build/protest_ARCH=vgg16_K=6_EPOCH=425_label_gaussianmixture.txt'


    random_visual_file(file)