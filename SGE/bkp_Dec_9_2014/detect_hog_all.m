baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/detectors_hog_s40';
inputData.inputDir = baseDir;

run_all(inputData,outDir,'detect_hog_parallel',false);