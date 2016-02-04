% clear all
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/res_s40_fisher_2';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'classify_parallel',true);
