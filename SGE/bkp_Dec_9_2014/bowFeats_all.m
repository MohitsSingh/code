clear all;
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/bow_s40_feats';
inputData.inputDir = baseDir;

run_all(inputData,outDir,'bowFeats_parallel',false);

load nonPersonIds;
fileList = struct('name',{});
for k = 1:length(nonPersonIds)
    fileList(k).name = [nonPersonIds{k} '.jpg'];
end
b = '/home/amirro/storage/VOCdevkit/VOC2011/JPEGImages/';
inputData.inputDir = baseDir;
inputData.fileList = fileList;
inputData.inputDir = b;
run_all(inputData,outDir,'bowFeats_parallel',false);