baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/occluders_s40_new4';
inputData.inputDir = baseDir;

run_all(inputData,outDir,'occluding_regions_parallel',false);