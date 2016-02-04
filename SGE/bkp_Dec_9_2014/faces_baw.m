function res = faces_baw(conf,I,reqInfo,moreParams)
if (nargin == 0)
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
conf_.get_full_image = false;
[I,I_rect] = getImage(conf_,I);
detections = struct('rot',{},'boxes',{});

if (isfield(moreParams,'resizeFactor'))
    resizeFactor = moreParams.resizeFactor;
else
    resizeFactor = 2;
end
if (isfield(moreParams,'rots'))
    rots = moreParams.rots;
else
    rots = 0;
end
I_orig = imresize(I,resizeFactor);

for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');
    [ds, bs] = imgdetect(I, model,-2);
    top = nms(ds, 0.1);
    detections(iRot).rot = rots(iRot);
    if (isempty(top))        
        detections(iRot).boxes = -inf(1,5);
    else    
        ds = ds(top(1:min(10,length(top))),:);
        ds(:,1:4) = ds(:,1:4)/resizeFactor+repmat(I_rect([1 2 1 2]),size(ds,1),1);
        detections(iRot).boxes = ds;
    end
end
res.detections = detections;