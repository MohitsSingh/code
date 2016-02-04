% baseDir = '/home/amirro/data/Stanford40/JPEGImages';
% outDir = '/home/amirro/storage/s40_keypoints_piotr';
% inputData.inputDir = baseDir;
% suffix =[];testing = true;
% load ~/code/mircs/faceActionImageNames;
%
% inputData.fileList = struct('name',faceActionImageNames);
%
% % run_all(inputData,outDir,'occluding_regions_parallel_2',false);
% run_all(inputData,outDir,'keypoints_piotr_parallel',testing ,suffix,'mcluster03');
%

%%
dfg
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_keypoints_piotr';
inputData.inputDir = baseDir;
suffix =[];testing = false;
load ~/code/mircs/faceActionImageNames;
inputData.fileList = struct('name',faceActionImageNames);
% load  ~/storage/misc/all_gloc_files
% inputData.fileList = struct('name',allFiles);
% run_all_2(inputData,outDir,fun,true,'','mcluster03')

run_all_2(inputData,outDir,'rcpr_parallel',testing ,suffix);
