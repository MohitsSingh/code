baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_sp_agglom_2_partial';
inputData.inputDir = baseDir;
checkIfNeeded = true;
suffix = [];
toDebug = false;
load ~/code/mircs/faceActionImageNames;
inputData.fileList = struct('name',faceActionImageNames);

run_all(inputData,outDir,'sp_agglom_parallel',toDebug,suffix,'mcluster03',checkIfNeeded);

