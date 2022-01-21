# newer than visualize_grid.py
# deals with deepcluster
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

cwd = "../study1_build/"

def visualize_single_file(file):
    basename = os.path.basename(file)

    sall = basename.split("_")
    pca = ""

    try:
        if basename.startswith("pca"):
            pca = "reduced-dimension"
            # dataset
            dataset = sall[1]

            # architecture
            architecture = sall[2].split("=")[-1]

            # K
            K = sall[3].split("=")[-1]
            K = int(K)

            # epoch
            epoch = sall[4].split("=")[-1]
        else:
            pca = "original"
            # dataset
            dataset = sall[0]

            # architecture
            architecture = sall[1].split("=")[-1]

            # K
            K = sall[2].split("=")[-1]
            K = int(K)

            # epoch
            epoch = sall[3].split("=")[-1]

    except:
        dataset = "politician"
        architecture = "VGG"
        K = 6
        epoch = 0
    # * read labeling ressults
    results = open(file, 'r').readlines()

    savefolder = "."


    ncol = 10 # how many images to show in each row
    nrow = 1
    nimg = ncol * nrow # total number of image in each cluster to show
    large_w = w_sq*ncol + w_clusterid # total width of the entire image
    large_h = (w_sq + h_imgid) * nrow * K # total height of the entire image
    large = Image.new('RGB', (large_w, large_h), "white")
    large_savepath = os.path.join(savefolder, dataset, feature_method, architecture + "_" + cluster_method + "_K=" + str(K)+ "_epoch=" + str(epoch)+ "_" + pca + ".png")

    def letter_range(start, stop="{", step=1):
        """Yield a range of lowercase letters.""" 
        for ord_ in range(ord(start.upper()), ord(stop.upper()), step):
            yield chr(ord_)

    # if os.path.exists(large_savepath):
    #     continue

    current_cluster = 0
    onecluster = []
    cluster_dict = defaultdict(list) # key is cluster assignment; value is filename

    # read clustering assignments: if we are using deepcluster or joint-cluster-cnn
    if feature_method == "pre_trained_feature" or feature_method == "deepcluster":
        for eachline in results:
            sall = eachline.split("\t")
            filename = sall[0]
            label = int(sall[1].strip())
            distance_to_centroid = float(sall[2].strip())

            cluster_dict[label].append((filename, distance_to_centroid))
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


    for label in range(0, K): 
        # print(f"---------- size of cluster {label}: {len(cluster_dict[label])}")
        # obtain image lists

        if len(cluster_dict[label]):

            # randomly sampling

            if random_order:
                imgfiles = cluster_dict[label][:nimg]
                # imgfiles = random.choices(cluster_dict[label], k = nimg)
                imgfiles = [x[0] for x in imgfiles]

            # select those closet to centroids
            else:
                closest_files = sorted(cluster_dict[label], key = lambda x: x[1])
                imgfiles = list(closest_files)[:nimg]
                imgfiles = [x[0] for x in imgfiles]



            imgfiles = [os.path.join(cwd,  x) for x in imgfiles]



            d = ImageDraw.Draw(large)
            d.text((0, label * nrow * (w_sq + h_imgid) + 15), str(label+1), font=fntcluster, fill=(0, 0, 0))


            for j, imgfile in enumerate(imgfiles):
                jw = j%ncol; jh = j//ncol
                img = Image.open(imgfile)
                im_sq = fill_square(img, w_sq)
                im_sq_x = jw*w_sq + w_clusterid
                im_sq_y = h_imgid + jh* (w_sq + h_imgid) + label * nrow * (w_sq + h_imgid)
                large.paste(im_sq, (im_sq_x, im_sq_y))

    print (d)
    rangea = list(letter_range("A", "Z"))
    for ylabel in range(0,ncol):

        # d.text((ylabel * nrow * (w_sq + h_imgid) + 18, 1), rangea[ylabel], font=fntcluster, fill=(0, 0, 0))
        d.text((ylabel*w_sq + w_clusterid + 25, 1), rangea[ylabel], font=fntcluster, fill=(0, 0, 0))

    # large_savepath = os.path.join(cwd, "img cluster", savefolder, str(K)+".png")

    if random_order:
        large_savepath = os.path.join(savefolder, architecture + "_" + cluster_method + "_K=" + str(K)+ "_epoch=" + str(epoch)+ "_" + pca + "_random_order.png")
    else:
        large_savepath = os.path.join(savefolder, architecture + "_" + cluster_method + "_K=" + str(K)+ "_epoch=" + str(epoch)+ "_" + pca + ".png")
    # large_savepath = os.path.join(cwd1, savefolder, method + "_" + str(K)+".png")
    print (large_savepath)
    #of.create_path(large_savepath)
    large.save(large_savepath)


if __name__ == '__main__':

    file = "protest_ARCH=vgg16_K=6_EPOCH=5_label_distance.txt"

    feature_method = "deepcluster"

    random_order = False

    visualize_single_file(file)
