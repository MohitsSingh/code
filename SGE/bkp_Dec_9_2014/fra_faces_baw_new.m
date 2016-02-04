function res = fra_faces_baw_new(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    addpath('~/code/mircs');
    config;
    load fra_db;
    res.conf = conf;
    cd /home/amirro/code/3rdparty/voc-release5
    startup
    load ~/code/3rdparty/dpm_baseline.mat
    res.model = model;
    return;
end
model = reqInfo.model;
conf_ = reqInfo.conf;

I = imread(I);
detections = struct('rot',{},'boxes',{});
I_orig = I;
%rots = -20:10:20;
rots = 0;
for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');
    [ds, bs] = imgdetect(I, model,-2);
    top = nms(ds, 0.1);
    if (isempty(top))
        boxes = -inf(1,5);
    end
    detections(iRot).rot = rots(iRot);
    detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
end
res.detections = detections;