
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
% outDir = '/home/amirro/storage/dpm_s40';
outDir = '/home/amirro/storage/dpm_s40_sun';
inputData.inputDir = baseDir;

run_all(inputData,outDir,'dpm_parallel',false);