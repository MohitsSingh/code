function parallel_helper_2(fun,params,outDir)
% runs a task in a parallel function : fun is applied to all inputs in d
% fun, when accepting no parameters should make any initializations
% necessary, and return data needed by all called to the function.
addpath('~/code/SGE');
addpath(genpath('~/code/utils'));
initData = feval(fun,'init');
for k = 1:length(params)
    curParams = params(k);
    curName = params.name;
    disp([curName '...']);
    drawnow('update');
    cachePath = j2m(outDir,curName);
    loadOrCalc_2(fun, initData, curParams,cachePath);
end
disp('done');