
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
% outDir = '/home/amirro/storage/dpm_s40';
outDir = '/home/amirro/storage/dpm_subclass_s40_2';
inputData.inputDir = baseDir;
run_all(inputData,outDir,'dpm_subclass_parallel',false);