

baseDir = '/home/amirro/data/Stanford40/JPEGImages';
d = dir(fullfile(baseDir,'*.jpg'));

outDir = '/home/amirro/storage/hands_s40_ita';
conf.handsDir = outDir;
ranges = {};

nSplits = 100;
d = d(randperm(length(d)));
for k = 0:nSplits-1
    ranges{k+1} = find(mod(1:length(d),nSplits)==k);
end

% create parallel code

code = 'cd /home/amirro/code/SGE; hands_parallel(baseDir,d,indRange,outDir)';
fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;
mkdir(outDir);
run_parallel(code, 'baseDir',baseDir,'d',d,'indRange',ranges,'outDir',outDir, '-cluster', 'mcluster03');
echo off;
fprintf('Done\n');

%% consolidate results.
cd /home/amirro/code/mircs
initpath;
config;
rmpath(genpath('/home/amirro/code/3rdparty/face-release1.0-basic/'));
addpath('/home/amirro/code/3rdparty/voc-release5');

%%
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
top_bbs_train = {};
b = 1:length(train_ids);
for r = 1:length(train_ids)
    k = b(r)
    handsFileName = fullfile(conf.handsDir,strrep(train_ids{k},'.jpg','.mat'));
    load(handsFileName)
%     pick  = esvm_nms(shape.boxes,.5);
    top_bbs_train{k} = boxes;
end

save ~/storage/hands_s40/top_bbs_train_ita.mat top_bbs_train

[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test',1,0);
top_bbs_test = {};
b = 1:length(test_ids);
for r = 1:length(test_ids)
    k = b(r)
    handsFileName = fullfile(conf.handsDir,strrep(test_ids{k},'.jpg','.mat'));
    load(handsFileName)
%     pick  = esvm_nms(shape.boxes,.5);
    top_bbs_test{k} = boxes;
end

save ~/storage/hands_s40/top_bbs_test_ita.mat top_bbs_test


% hands_parallel(baseDir,d,1,outDir)