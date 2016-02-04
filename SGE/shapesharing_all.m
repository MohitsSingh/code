baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_shape_sharing';
inputData.inputDir = baseDir;
checkIfNeeded = true;
suffix = [];
toDebug = false;
load ~/code/mircs/faceActionImageNames;
inputData.fileList = struct('name',faceActionImageNames);
run_all(inputData,outDir,'shapesharing_parallel',toDebug,suffix,'mcluster03',checkIfNeeded);

%%
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '~/storage/s40_shape_sharing_face';
inputData.inputDir = baseDir;
checkIfNeeded = true;
suffix = [];
toDebug = true;
load ~/code/mircs/faceActionImageNames;
inputData.fileList = struct('name',faceActionImageNames);
run_all(inputData,outDir,'shapesharing_parallel_face_only',toDebug,suffix,'mcluster03',checkIfNeeded);