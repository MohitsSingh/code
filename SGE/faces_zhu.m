function res = faces_zhu(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    %     addpath('~/code/mircs');
    %     cd ~/code/mircs
    
%     initpath
%     config
    res = [];
    addpath(genpath('~/code/utils'));
    addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
    cd /home/amirro/code/3rdparty/face-release1.0-basic/
    return
end
I = params.img;
% I = imResample(params.img,[128 128],'bilinear');
res.landmarks = extractLandmarks(I,-20:20:20,{'face_p146_small'});

