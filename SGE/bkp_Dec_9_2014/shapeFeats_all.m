baseDir = '/home/amirro/data/Stanford40/JPEGImages';
d = dir(fullfile(baseDir,'*.jpg'));
outDir = '/home/amirro/storage/shape_s40';
ranges = {};

nSplits = 100;
d = d(randperm(length(d)));
for k = 0:nSplits-1
    ranges{k+1} = find(mod(1:length(d),nSplits)==k);
end

% create parallel code

code = 'cd /home/amirro/code/SGE; shape_parallel(baseDir,d,indRange,outDir)';
fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;
mkdir(outDir);
run_parallel(code, 'baseDir',baseDir,'d',d,'indRange',ranges,'outDir',outDir,...
    '-cluster', 'mcluster03');
echo off;
fprintf('Done\n');

% %shape_parallel(baseDir,d,1:1000,outDir)

