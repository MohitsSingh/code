function [model,locked] = runDPMLearning(cls, n, trainSet, valSet,override)
if (nargin < 5)
    override = false;
end
currentDir = pwd;
dpmDir = '/home/amirro/code/3rdparty/voc-release5';
cd(dpmDir);
modelPath = ['models/' cls '_final.mat'];
lockPath = [modelPath '.dummy'];
model = [];
locked = false;

if (exist(modelPath,'file'))
    load(modelPath);
    cd(currentDir);
    return;
end


if (~override && exist(lockPath,'file'))
    locked = true;
    cd(currentDir);
    return;
end

fclose(fopen(lockPath,'w'));
% if (exist(modelPath,'file'))
%     load(modelPath);
%     cd(currentDir);
%     return;
% end
% save tmp.mat cls n trainSet valSet currentDir;
% clear all;
% load tmp.mat;
model = amir_train_script(cls, n, trainSet, valSet);
cd(currentDir);
end

%%