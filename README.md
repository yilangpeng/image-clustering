
This repository contains the replication files for article "Image Clustering: An Unsupervised Approach to Categorize Visual Data in Social Science Research."

# Data access
This project uses three datasets. Study 1's dataset can be accessed at this [link](https://doi.org/10.7910/DVN/VSOH5H). Study 2 uses two datasets. The first dataset can be accessed via the same link provided before. You can also download the images directly from Instagram. The second dataset can be requested directly from Twitter [link](https://transparency.twitter.com/en/reports/information-operations.html). Please choose the batch released in October 2018.

# Study 1:
-  study1_build: extract intermediate low-dimensional vector representation of Study 1(China Protest) and store the features and labels. The key folders and files are:
  - images/: 14,127 images used in Study 1 of the manuscript.
  - bovw/: bag-of-visual-word model. To replicate:
    - first run `extract_bbow_features.py` :extract intermediate low-dimensional vector representation and save it to `protest_bbow.csv`
  - self-supervised/ : self-supervised algorithm based on DeepCluster (Caron et al., 2018). The code was based on the original coding implmenetation at https://github.com/facebookresearch/deepcluster
    - Run these two files. Note that when running these two files, it will first download several large pre-trained models (over GB) and then save the training results as checkpoints. So you should have at least 5 GB remaining on your disk to successfully run these scripts. The two .sh files will save extracted features in files with extension .pickle. These are binary format vectors
      - `train_self_supervised_from_scratches.sh`: extract intermediate low-dimensional vector representation using Deep Cluster from scratch.
      - `train_self_supervised_transfer.sh`: use the DeepCluster but did not train the model; just use their model to extract intermediate features, hence transfer learning.
  - supervised/ : transfer learning based on VGG and ImageNet dataset. 
    - run `extract_features.py` to extract  intermediate low-dimensional vector representation. The extracted features will be saved to "img transform/" folder.
  - `clustering.py` : this python script will run clustering algorithms over all the previous extracted intermediate representations. 
- Fig1/ ; Fig 4/ ; .... All these folders contain one python script. Run the script and you will get corresponding figure used in the manuscript.
--------------------------------------
Figures we we cannot replicate now:
- Fig 1 (purely random)
- Fig 4 (bag of visual words)
- FigB5 (resnet)

# Study 2A
## A1/2/3 feature extraction.py
For each image, this script extracts features from a pre-trained model and saves features in the "img exfeature" folder. If you want to use the places365 model, please download the model from this [link](https://github.com/GKalliatakis/Keras-VGG16-places365) and place all the relevant scripts in the same folder. 

## B combine features.py
This script combines all the features into one file.

## C PCA.py
This script conducts principal component analysis on the extract features.

## D kmeans clustering.py
This script applies k-means clustering to the first 200 dimensions in PCA, with the number of clusters ranging from 6, 8, to 10.

## E copy image.py
For each cluster in each clustering solution, this script randomly selects 20 images and copies them to the "img cluster" folder.

## F visualize grid.py
For each clustering solution, this script creates a figure that show the randomly selected 20 images in each cluster.

# Study 2B
The scripts are similar to the ones in study 2A.

