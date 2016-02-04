% function sge_gpb(imgs,baseDir)
%%sge_gpb

% multiWrite(imgs,baseDir);
baseDir = '/home/amirro/data/Stanford40/JPEGImages';
outDir = '/home/amirro/storage/gpb_s40';
%run_all(baseDir,outDir,'gpb_parallel',true,'_regions.mat');
inputData.inputDir = baseDir;
% run_all(inputData,outDir,'gpb_parallel',false);



outDir = '~/storage/gpb_s40_face_2';
for k = 1:100
    system('rm ~/sge_parallel_new/*');
    run_all(inputData,outDir,'gpb_parallel_new_face_only',false);
end

load nonPersonIds;
fileList = struct('name',{});
for k = 1:length(nonPersonIds)
    fileList(k).name = [nonPersonIds{k} '.jpg'];
end
b = '/home/amirro/storage/VOCdevkit/VOC2011/JPEGImages/';
inputData.fileList = fileList;
inputData.inputDir = b;

run_all(inputData,outDir,'gpb_parallel',false,'_regions.mat');

% d = dir(fullfile(baseDir,'*.jpg'));

% filenames = {};
% for k = 1:length(d)
%     filenames{k} = fullfile(baseDir,d(k).name);
% end
% 
% ranges = {};
% 
% nSplits = 100;
% 
% for k = 0:nSplits-1
%     ranges{k+1} = find(mod(1:length(d),nSplits)==k);
% end
% 
% % create parallel code
% 
% code = 'cd /home/amirro/code/SGE; gpb_parallel(baseDir,d,indRange,outDir)';
% fprintf('Parallel code:\n%s\n', code);
% 
% % run parallel code
% fprintf('Running in parallel:\n');
% echo on;
% outDir = '/home/amirro/storage/gpb_s40';
% run_parallel(code, 'baseDir',baseDir,'d',d,'indRange',ranges,'outDir',outDir, '-cluster', 'mcluster01');
% echo off;
% fprintf('Done\n');
% 
% % gpb_parallel(baseDir,d,1:1000,outDir)
% 
