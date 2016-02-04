function run_all_2(inputData,outDir,fun,justTesting,suffix,cluster_name,checkIfNeeded,moreParams)
if (nargin < 5 || isempty(suffix))
    suffix = '.mat';
end
if (nargin < 6 || isempty(cluster_name))
    cluster_name = 'mcluster03';
end
if (nargin < 7)
    checkIfNeeded = false;
end
if (nargin < 8)
    moreParams = [];
end
% if (nargin < 7)
%     onlyFaceActions = false;
% end

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
    
    for t = 1:length(d)
        d(t).name = fullfile(inputDir,d(t).name);
    end
    
end

if (checkIfNeeded)
    dd = dir(fullfile(outDir,['*' suffix]));
    sizes = [dd.bytes];
    
%       for u = 1:length(dd)
%         if (dd(u).bytes < 150000)
%             u
%             delete(fullfile(outDir,dd(u).name));
%         end
%     end
    dd = {dd.name};
  
    
    d1 ={};
    for t = 1:length(d)
        d1{t} = strrep(d(t).name,'.jpg',suffix);
    end
    
    d = struct('name',setdiff(d1,dd));
    for t = 1:length(d)
        d(t).name = strrep(d(t).name,suffix,'.jpg');
    end
end
% if (checkIfNeeded)
%     needsWork = false(size(d));
%     for k = 1:length(d)
%             k
%         if (~exist(fullfile(outDir,strrep(d(k).name,'.jpg',suffix)),'file'))
%             needsWork(k) = true;
%         end
%     end
%     d = d(needsWork);
% end
ranges = {};
nSplits = 100; % TODO - was 100
d = d(randperm(length(d)));
for k = 0:nSplits-1
    ranges{k+1} = find(mod(1:length(d),nSplits)==k);
end
ranges = ranges(cellfun(@any,ranges));
mkdir(outDir);
if (~exist('moreParams','var'))
    moreParams = [];
end
% create parallel code
if (justTesting)
    parallel_helper(fun,d,1:length(d),outDir,moreParams);
    %     feval(fun,inputDir,d,1:length(d),outDir);
    return
end
% code = ['cd /home/amirro/code/SGE; ' fun '(inputDir,d,indRange,outDir)'];

code = ['cd /home/amirro/code/SGE; ' 'parallel_helper(fun,d,indRange,outDir,moreParams)'];

fprintf('Parallel code:\n%s\n', code);

% run parallel code
fprintf('Running in parallel:\n');
echo on;

run_parallel(code, 'fun',fun,'inputDir',inputDir,'d',d,'indRange',ranges,'outDir',outDir,...
    'moreParams',moreParams, '-cluster', cluster_name);
echo off;
fprintf('Done\n');

end