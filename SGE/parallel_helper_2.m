function parallel_helper_2(fun,params,outDir)
% runs a task in a parallel function : fun is applied to all inputs in
% params. params is a struct array where each element describes parameters
% for a single call of fun.
% fun , when accepting no parameters should make any initializations
% necessary, and return data needed by subsequent calls.
addpath('~/code/SGE');
addpath(genpath('~/code/utils'));
initData = feval(fun,'init');
for k = 1:length(params)
    curParams = params(k);
    curName = curParams.name;
    disp([curName '...']);
    drawnow('update');
    [~,name,ext] = fileparts(curName);
    curName = [name ext];
    cachePath = j2m(outDir,curName);
    loadOrCalc(fun, initData, curParams,cachePath);
end
disp('done');