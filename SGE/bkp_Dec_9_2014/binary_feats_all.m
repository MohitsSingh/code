inputData.inputDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/relativeFeats_s40';
mkdir(outDir);
run_all(inputData,outDir,'get_binary_feats_parallel',false);