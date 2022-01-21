

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
