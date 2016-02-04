#!/bin/sh

# MODIFY PATH for YOUR SETTING

CAFFE_DIR=/home/beny/code/3rd/deepLab/deeplab-deeplab-public-3e413eed0de8
CAFFE_BIN=${CAFFE_DIR}/.build_release/tools/caffe.bin

NUM_LABELS=21
DATA_ROOT=/home/amirro/storage/data/Stanford40/ppm512x512/
EXP=voc12
EXP1=voc12/ # lists of images is under $EXP/list/

NET_ID=vgg128_noup
DEV_ID=0

# Run

for TEST_SET in test_images; do
    TEST_ITER=`cat $EXP1/list/${TEST_SET}.txt | wc -l`
    MODEL=${CAFFE_DIR}/models/DeepLab-MSc-COCO-LargeFOV/train2_iter_8000.caffemodel
    echo Testing2 net
    FEATURE_DIR=/home/amirro/code/3rdparty/my_deeplab_scripts/workdir/
    CFG_DIR=$EXP1 # ${CAFFE_DIR}/models/DeepLab-MSc-COCO-LargeFOV
    mkdir -p ${FEATURE_DIR}/${TEST_SET}/fc8
    mkdir -p ${FEATURE_DIR}/${TEST_SET}/crf
    sed "$(eval echo $(cat /home/beny/code/visualQuestionAnswering/deepLabWorkDir/sub.sed))" \
	${CFG_DIR}/test.prototxt > ${CFG_DIR}/test_${TEST_SET}.prototxt
    CMD="${CAFFE_BIN} test \
         --model=${CFG_DIR}/test_${TEST_SET}.prototxt \
         --weights=${MODEL} \
         --gpu=${DEV_ID} \
         --iterations=${TEST_ITER}"
    echo Running ${CMD} && ${CMD}
done

