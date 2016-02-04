

% length of region_data_train : 4003
% length of region_data_test : 5532

outDir = '~/storage/s40_region_features';
code = ['cd /home/amirro/code/SGE;' 'region_features_parallel(indRange,outDir,isTrain)'];
fprintf('Parallel code:\n%s\n', code);
% mkdir(outDir)
% run parallel code
fprintf('Running in parallel:\n');
echo on;

ranges_train = {};
nSplits = 100; % TODO - was 100
% d = d(randperm(length(d)));
T = 4100;
R = 1:floor(T/nSplits):T;
R(end) = T+1;

for q = 1:length(R)-1
    ranges_train{q} = R(q):R(q+1)-1;
end

run_parallel(code, 'indRange',ranges_train,'outDir',outDir,'isTrain',true,...
    '-cluster', 'mcluster01');

T = 5600;
R = 1:floor(T/nSplits):T;
R(end) = T+1;
ranges_test = {};
for q = 1:length(R)-1
    ranges_test{q} = R(q):R(q+1)-1;
end

run_parallel(code, 'indRange',ranges_test,'outDir',outDir,'isTrain',false,...
    '-cluster', 'mcluster01');


% collect the results.
d = dir(fullfile(outDir,'*train.mat'));
feats_train = {};
for k = 1:length(d)
    k
    dd = load(fullfile(outDir,d(k).name));
    if (~isempty(dd.feats))
        feats_train{end+1} = dd.feats;
    end
end

feats_train = cat(2,feats_train{:});
save /home/amirro/mircs/experiments/experiment_0015/feats_train2.mat feats_train

d = dir(fullfile(outDir,'*test.mat'));
feats_test = {};
for k = 1:length(d)
    k
    dd = load(fullfile(outDir,d(k).name));
    if (~isempty(dd.feats))
        feats_test{end+1} = dd.feats;
    end
end

feats_test = cat(2,feats_test{:});
save /home/amirro/mircs/experiments/experiment_0015/feats_test2.mat feats_test
