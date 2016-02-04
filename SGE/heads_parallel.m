function res = heads_parallel(conf,imagePath,reqInfo)
if (nargin == 0)
    addpath('~/code/SGE/');
    cd /home/amirro/code/3rdparty/voc-release4.01;
    startup;
    addpath('laeohead_v2');
    addpath(genpath('~/code/utils'));
    load('head-gen-on-ub-4laeo.mat'); % IJCV'2013
    res.model = model;
    res.detthr = -1; % was -.82
    res.conf = [];
    %     lsymbs = '><><><';
    return;
end

model = reqInfo.model;
detthr = reqInfo.detthr;
img = imread(imagePath);
%% Run head detector on the whole image: it works better on upper-body areas.
[dets, boxes] = imgdetect(img, model, -1.5);
if (isempty(dets))
    res = [];
else
    % Keep only the best ones after NMS
    top = nms(dets, 0.5);
    dets = dets(top,:);
    res = dets(:,[1:4 end-1 end]);
end
end