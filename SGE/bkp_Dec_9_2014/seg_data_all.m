% clear all
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_seg_data';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/faceActionImageNames;
inputData.fileList = struct('name',faceActionImageNames(isValid(img_sel)));
run_all(inputData,outDir,'seg_data_parallel',testing ,suffix,'mcluster03');
