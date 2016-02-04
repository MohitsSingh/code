function res = run_and_collect_results(inputs,fun,justTesting,extraInfo,job_suffix,nJobs,outDir)
% delete old jobs
pp = pwd;
cd('~/sge_parallel_new/');
system('find -mmin +60 -exec rm {} +');
cd(pp);
res = [];
if (nargin < 3)
    justTesting = false;
end
if (nargin < 4)
    extraInfo = [];
end
if (nargin < 5)
    job_suffix = [];
end
if (nargin < 6 || isempty(nJobs))
    nJobs = 50;
end
if (nargin < 7)
    outDir = fullfile('~/storage/tmp/',job_suffix);
end
ensuredir(outDir);
[data,inds] = splitToRanges(inputs,nJobs);
jobIDS = mat2cell2(1:length(data),[1,length(data)]);
% delete(fullfile(outDir,['*' job_suffix '_agg.mat']));

% create parallel code
if (justTesting)
    feval(fun,data{1},inds{1},jobIDS{1},outDir,extraInfo,job_suffix);
    return
end
code = ['cd /home/amirro/code/SGE; ' fun '(data,inds,jobID,outDir,extraInfo,jobSuffix)'];
fprintf('Parallel code:\n%s\n', code);
% run parallel code
fprintf('Running in parallel:\n');
echo on;
%imagePaths,imageInds,jobID,outDir,extraInfo
run_parallel(code, 'data',data,'inds',inds,'jobID',jobIDS,'outDir',outDir,'extraInfo',extraInfo,'jobSuffix',job_suffix,...
    '-cluster', 'mcluster03');

d = dir(fullfile(outDir,['*' job_suffix '_agg.mat']));

all_res = {};
all_inds = {};
for k = 1:length(d)
    r = load(fullfile(outDir,d(k).name));
    all_res{k} = r.res(:);
    all_inds{k} = r.inds(:);
end
delete(fullfile(outDir,['*' job_suffix '_agg.mat']));

res = cat(1,all_res{:});
inds = cat(1,all_inds{:});
[~,is] = sort(inds,'ascend');
res = res(is);

echo off;
fprintf('Done\n');
end