baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/s40_heads';
inputData.inputDir = baseDir;
suffix =[];testing = false;
%load ~/code/mircs/faceActionImageNames;

% faceActionImageNames = {'blowing_bubbles_077.jpg'};
% for k = 1:length(faceActionImageNames)
%farceActionImageNames{k} = fullfile(baseDir,faceActionImageNames{k});
% end
% inputData.fileList = struct('name',faceActionImageNames);
% run_all(inputData,outDir,'occluding_regions_parallel_2',false);
inputPaths = {};
d = dir(fullfile(baseDir,'*.jpg'));
for k = 1:length(d)
    inputPaths{k} = fullfile(baseDir,d(k).name);
end
inputData.fileList = struct('name',inputPaths);
run_all_2(inputData,outDir,@heads_parallel,testing ,suffix,'mcluster03');