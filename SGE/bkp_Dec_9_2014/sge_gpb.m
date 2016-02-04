function sge_gpb(imgs,baseDir)

% multiWrite(imgs,baseDir);
d = dir(fullfile(baseDir,'*.tif'));
% filenames = {};
% for k = 1:length(d)
%     filenames{k} = fullfile(baseDir,d(k).name);
% end

ranges = {};

nSplits = 100;

for k = 0:nSplits-1
    ranges{k+1} = find(mod(1:length(d),nSplits)==k);
end

% create parallel code
code=  'cd /home/amirro/code/mircs; calculateGpbParaller(baseDir,d,indRange)';
fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;
run_parallel(code, 'baseDir',baseDir,'d',d,'indRange',ranges, '-cluster', 'mcluster01');
echo off;
fprintf('Done\n');
