import codecs, os, sys, random, shutil, string, glob
from PIL import Image, ImageDraw, ImageFont
from natsort import natsorted
import numpy as np
import ypoften as of

def fill_square(im, tbw):
    size = (tbw, tbw)
    bg = Image.new('RGB', size, "white")  # create a background image  
    im.thumbnail(size, Image.ANTIALIAS)
    w, h = im.size 
    bg.paste(im, (int((size[0]-w)/2), int((size[1]-h)/2)))  
    return(bg)

cvmodels = ["vgg16 fc1","vgg16 places fc1","vggface fc6"]

cwd = '/Volumes/Cuba/Analysis/IRA/Github/'

for cvmodel in cvmodels:
    features_array = []

    savefolder = cvmodel + " " + clmethod + " PCA200"

    w_sq = 300 # width/height of image
    w_clusterid = 60 # width of text for cluster id
    h_imgid = 60 # height of gap
    fntcluster = ImageFont.truetype('/Library/Fonts/Arial.ttf', 60)
    
    for K in [6,8,10]:
        ncol = 20 # how many images to show in each row
        nrow = 1
        nimg = ncol * nrow # total number of image in each cluster to show
        large_w = w_sq*ncol + w_clusterid # total width of the entire image
        large_h = (w_sq + h_imgid) * nrow * K # total height of the entire image
        large = Image.new('RGB', (large_w, large_h), "white")
    
        for label in range(0, K): 
            print('-'* 10, K, label)
            imgpathfolder = os.path.join(cwd,'img cluster',savefolder,str(K),str(label),'')
            imgfiles = glob.glob(imgpathfolder + "*.png")
            imgfiles = natsorted(imgfiles)[:nimg]
    
            d = ImageDraw.Draw(large)
            d.text((0, label * nrow * (w_sq + h_imgid) + 15), str(label+1), font=fntcluster, fill=(0, 0, 0))
    
            for j, imgfile in enumerate(imgfiles):
                jw = j%ncol; jh = j//ncol
                img = Image.open(imgfile)
                im_sq = fill_square(img, w_sq)
                im_sq_x = jw*w_sq + w_clusterid
                im_sq_y = h_imgid + jh* (w_sq + h_imgid) + label * nrow * (w_sq + h_imgid)
                large.paste(im_sq, (im_sq_x, im_sq_y))
    
        large_savepath = os.path.join(cwd, "img cluster", 'grid', savefolder, str(K)+".png")
        of.create_path(large_savepath)
        large.save(large_savepath)
    
    print("DONE"*20)