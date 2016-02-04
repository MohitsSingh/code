function res = fra_faces_baw(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    initpath;
    addpath('~/code/mircs');
    config;
    load fra_db;
    res.fra_db = fra_db;
    res.conf = conf;
    cd /home/amirro/code/3rdparty/voc-release5
    startup
    load ~/code/3rdparty/dpm_baseline.mat
    res.model = model;    
    return;
end
model = reqInfo.model;
conf_ = reqInfo.conf;
fra_db = reqInfo.fra_db;

if (nargin == 4)
    fra_db = moreParams.fra_db;
    face_only = moreParams.face_only;
else
    face_only =false;
end

if (any(strfind(I,'aflw_cropped_context')))
    I = imread(I);
else
    k = findImageIndex(fra_db,I);
    curImageData = fra_db(k);
    curImageData = fra_db(k);
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    roiParams.centerOnMouth = false;
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf_,curImageData,roiParams);
    res.roiBox = roiBox;
    res.roiParams = roiParams;
    
end
detections = struct('rot',{},'boxes',{});
I_orig = I;
rots = -40:10:40;
% rots = 0;
for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');
    [ds, bs] = imgdetect(I, model,0);
    top = nms(ds, 0.1);
    if (isempty(top))
        boxes = -inf(1,5);
    end
    detections(iRot).rot = rots(iRot);
    detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
end
res.detections = detections;