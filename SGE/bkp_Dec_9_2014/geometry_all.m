clear all;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/geometry_s40';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'geometry_parallel',true);