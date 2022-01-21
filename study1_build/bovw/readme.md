Scripts:

|____extract_bbow_features.py : extract vector representations of protest images and store in file "protest_bbow.csv"
|____kmeans.py: take file "protest_bbow.csv" as input and ouput clustering labels as results (e.g., "_protest_bbow_K=6_label_distance.txt")

Data: 
|____protest_bbow.csv: extracted features. 
|____protest_bbow_K=6_label_distance.txt:  Extracted labels. Each line begins with the filename, then the label assignment, and then the distance to the centroid of each cluster. Separated by tab.
