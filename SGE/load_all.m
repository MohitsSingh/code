function [res,goods] = load_all(fun,params,outDir,skipNonExisting)
% aggregates all results from parallel run.
% params. params is a struct array where each element describes parameters
% for a single call of fun.
% fun , when accepting no parameters should make any initializations
% necessary, and return data needed by subsequent calls.

if nargin < 4
    skipNonExisting = false;
end

addpath(genpath('~/code/utils'));
id = ticStatus('loading results...',.5,.5);
res = {};
goods = false(size(params));
for k = 1:length(params)
    curParams = params(k);
    curName = curParams.name;
    [~,name,ext] = fileparts(curName);
    curName = [name ext];
    %disp([curName '...']);
    cachePath = j2m(outDir,curName);
    if skipNonExisting && ~exist(cachePath,'file')
        continue
    end
    goods(k) = true;
    res {k} = loadOrCalc(fun, [], curParams,cachePath);
    tocStatus(id,k/length(params));
end
disp('done');