% clear all
inputData.inputDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/lineseg_s40';

run_all(inputData,outDir,'lineseg_parallel',false);