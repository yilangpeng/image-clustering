# Han Zhang
# modified on main.sh base don the original deepcluster GitHub repo
#
#!/bin/bash


DIR="/home/han/Dropbox/Collaborations/with_Yilang_Peng/2019-03-06-protest-features"
LR=0.05
WD=-5
WORKERS=16 ## change this to the number of CPU on your own machine
PYTHON="/home/han/miniconda3/bin/python"
EXP="exp/"
SAVEFOLDER="."


for data_name in "protest"; do
	for K in 6; do

		for ARCH in "vgg16" "alexnet"; do
			if [ $ARCH == "vgg16" ]; then
				EPOCHS=426 ## this is where the pre-trained model stopped
				BATCH=16 ## vgg model is more compelx so the batch number should be smaller than that of AlexNet
				## this is the pre-trained model location;
				## needs to download by yourself (it's quite big)
				## and then 
				TRAINED="/home/han/deepcluster_models/vgg16/checkpoint.pth.tar" 
			fi

			if [ $ARCH == "alexnet" ]; then

				EPOCHS=501 # where 
				BATCH=128
				TRAINED="/home/han/deepcluster_models/alexnet/checkpoint.pth.tar"
			fi

			CUDA_VISIBLE_DEVICES=1 ${PYTHON} main.py ${DIR} --name ${data_name} --savefolder "${SAVEFOLDER}" --exp ${EXP} --arch ${ARCH} --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS} --batch ${BATCH} --resume ${TRAINED} --epochs ${EPOCHS}
		done
	done
done
