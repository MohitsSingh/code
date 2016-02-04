function parallel_helper(fun,d,indRange,outDir,moreParams)
% runs a task in a parallel function : fun is applied to d(indrange)
% fun, when accepting no parameters should make any initializations
% necessary, and return data needed by all called to the function.
addpath('~/code/SGE');
addpath(genpath('~/code/utils'));
reqInfo = feval(fun);
if (nargin < 5)
    moreParams = [];
end
for k = 1:length(indRange)
    curPath = d(indRange(k)).name;
    disp([curPath '...']);
    drawnow('update');
    cachePath = j2m(outDir,curPath);
    if (~isfield(reqInfo,'conf'))
        reqInfo.conf = [];
    end
    loadOrCalc(reqInfo.conf,fun,curPath,cachePath,reqInfo,moreParams);
end
disp('done');