#!/bin/bash 

###########################################
# You can either use this script to generate the DenseCRF post-processed results
# or use the densecrf_layer (wrapper) in Caffe
###########################################

# specify the parameters
MAX_ITER=10

Bi_W=5     
Bi_X_STD=50
Bi_Y_STD=50
Bi_R_STD=3 
Bi_G_STD=3 
Bi_B_STD=3 

POS_W=3
POS_X_STD=3
POS_Y_STD=3


#######################################
# MODIFY THE PATY FOR YOUR SETTING
#######################################
SAVE_DIR=/home/amirro/code/3rdparty/matconvnet-fcn-master/res_W${Bi_W}_XStd${Bi_X_STD}_RStd${Bi_R_STD}_PosW${POS_W}_PosXStd${POS_X_STD}

echo "SAVE TO ${SAVE_DIR}"

CRF_DIR=/home/beny/code/3rd/deepLab/deeplab-deeplab-public-3e413eed0de8/densecrf

# NOTE THAT the densecrf code only loads ppm images
IMG_DIR=/home/amirro/code/3rdparty/matconvnet-fcn-master/d1/

# the features are saved in .mat format
CRF_BIN=${CRF_DIR}/prog_refine_pascal_v4
FEATURE_DIR=/home/amirro/code/3rdparty/matconvnet-fcn-master/feats/

mkdir -p ${SAVE_DIR}

# run the program
${CRF_BIN} -id ${IMG_DIR} -fd ${FEATURE_DIR} -sd ${SAVE_DIR} -i ${MAX_ITER} -px ${POS_X_STD} -py ${POS_Y_STD} -pw ${POS_W} -bx ${Bi_X_STD} -by ${Bi_Y_STD} -br ${Bi_R_STD} -bg ${Bi_G_STD} -bb ${Bi_B_STD} -bw ${Bi_W}

