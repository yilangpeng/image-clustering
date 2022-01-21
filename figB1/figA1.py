import codecs, os, glob
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import numpy as np
from random import sample 
import random
from collections import defaultdict

def letter_range(start, stop="{", step=1):
    """Yield a range of lowercase letters.""" 
    for ord_ in range(ord(start.upper()), ord(stop.upper()), step):
        yield chr(ord_)

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


# number of clusters
K = 5

dataset = ""

# feature_method = "deepcluster"
#feature_method = "joint-cluster-cnn"
feature_method = "pre_trained_feature"


# Architecture used
architecture = "VGG16"


# clustering method
cluster_method = "KMeans" 
cluster_method = "Hierarchical" 


cwd = "../study1_build/"


savefolder = "."

results = open("/home/han/Dropbox/Collaborations/with_Yilang_Peng/YP protest/img clustering/labels/protest_ARCH=vgg16_K=6_EPOCH=425_hierarchical.txt", 'r').readlines()


for K in range(6,7):
    ncol = 10 # how many images to show in each row
    nrow = 1
    nimg = ncol * nrow # total number of image in each cluster to show
    large_w = w_sq*ncol + w_clusterid # total width of the entire image
    large_h = (w_sq + h_imgid) * nrow * K # total height of the entire image
    large = Image.new('RGB', (large_w, large_h), "white")


    current_cluster = 0
    onecluster = []
    cluster_dict = defaultdict(list) # key is cluster assignment; value is 

    # read clustering assignments: if we are using deepcluster or joint-cluster-cnn
    if feature_method == "pre_trained_feature":
        for eachline in results:
            sall = eachline.split("\t")
            filename = sall[0]
            label = int(sall[1].strip())
            cluster_dict[label].append(filename)
    else:
        for eachline in results:
            if eachline.startswith("Cluster"):
                if len(onecluster) == 0:
                    continue
                else:
                    cluster_dict[current_cluster] = onecluster
                    onecluster = []
                    current_cluster += 1
                    continue
            elif eachline.startswith("-----"):
                continue
            else:
                sall = eachline.strip().split()[1:]
                onecluster.append(" ".join(sall))
    # print (cluster_dict.keys())
    # for each in cluster_dict.keys():
    #     print (each, len(cluster_dict[each]))

    for label in range(0, K): 
        print('-'* 10, "number of cluster:", K, label)
        # obtain image lists
        imgfiles = [os.path.join(cwd,  x) for x in cluster_dict[label]]

        imgfiles = random.choices(imgfiles, k = nimg)


        d = ImageDraw.Draw(large)
        d.text((0, label * nrow * (w_sq + h_imgid) + 15), str(label+1), font=fntcluster, fill=(0, 0, 0))

        for j, imgfile in enumerate(imgfiles):
            jw = j%ncol; jh = j//ncol
            img = Image.open(imgfile)
            im_sq = fill_square(img, w_sq)
            im_sq_x = jw*w_sq + w_clusterid
            im_sq_y = h_imgid + jh* (w_sq + h_imgid) + label * nrow * (w_sq + h_imgid)
            large.paste(im_sq, (im_sq_x, im_sq_y))

    rangea = list(letter_range("A", "Z"))
    for ylabel in range(0,ncol):

        # d.text((ylabel * nrow * (w_sq + h_imgid) + 18, 1), rangea[ylabel], font=fntcluster, fill=(0, 0, 0))
        d.text((ylabel*w_sq + w_clusterid + 25, 1), rangea[ylabel], font=fntcluster, fill=(0, 0, 0))
        
    print (large)
    # large_savepath = os.path.join(cwd, "img cluster", savefolder, str(K)+".png")
    large_savepath = os.path.join(savefolder, dataset, architecture + "_" + cluster_method + "_K=" + str(K)+".png")
    # large_savepath = os.path.join(cwd1, savefolder, method + "_" + str(K)+".png")
    print (large_savepath)
    # of.create_path(large_savepath)
    large.save(large_savepath)

