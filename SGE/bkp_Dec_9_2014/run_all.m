function run_all(inputData,outDir,fun,justTesting,suffix,cluster_name,checkIfNeeded)
if (nargin < 5 || isempty(suffix))
    suffix = '.mat';
end
if (nargin < 6 || isempty(cluster_name))
    cluster_name = 'mcluster03';
end
if (nargin < 7)
    checkIfNeeded = false;
end
if (nargin < 7)
    onlyFaceActions = false;
end

% load nonPersonIds;
% d = struct('name',{});
% for k = 1:length(nonPersonIds)
%     d(k).name = [nonPersonIds{k} '.jpg'];
% end
inputDir = inputData.inputDir;
if (isfield(inputData,'fileList'))
    
    d = inputData.fileList;
elseif (isfield(inputData,'images'))
    d = struct('img',{});
    for k = 1:length(inputData.images)
        d(k).img = inputData.images{k};
    end
    %     d = inputData.images;
else
    d = dir(fullfile(inputDir,'*.jpg'));
    %     d = dir(fullfile(inputDir,'drinking*.jpg'));
    %     d = dir(fullfile(inputDir,'blowing_bubbles_060.jpg'))
    %     d = dir(fullfile(inputDir,'*3000_000093.jpg'));
end

if (checkIfNeeded)
    needsWork = false(size(d));
    for k = 1:length(d)
        %     k
        if (~exist(fullfile(outDir,strrep(d(k).name,'.jpg',suffix)),'file'))
            needsWork(k) = true;
        end
    end
    d = d(needsWork);   
end
ranges = {};
nSplits = 100; % TODO - was 100
d = d(randperm(length(d)));
for k = 0:nSplits-1
    ranges{k+1} = find(mod(1:length(d),nSplits)==k);
end
ranges = ranges(cellfun(@any,ranges));
mkdir(outDir);

% create parallel code
if (justTesting)
    feval(fun,inputDir,d,1:length(d),outDir);
    return
end
code = ['cd /home/amirro/code/SGE; ' fun '(inputDir,d,indRange,outDir)'];
fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;

run_parallel(code, 'inputDir',inputDir,'d',d,'indRange',ranges,'outDir',outDir,...
    '-cluster', cluster_name);
echo off;
fprintf('Done\n');

end