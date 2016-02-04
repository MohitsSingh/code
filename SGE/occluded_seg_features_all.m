% clear all
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/occluded_seg_features_s40';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'occluded_seg_features_parallel',false);
