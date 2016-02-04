baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/occluders_s40_new_6';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/faceActionImageNames;

% faceActionImageNames = {'blowing_bubbles_077.jpg'};
inputData.fileList = struct('name',faceActionImageNames);

% run_all(inputData,outDir,'occluding_regions_parallel_2',false);
run_all(inputData,outDir,'occluding_regions_parallel_2',testing ,suffix,'mcluster03');