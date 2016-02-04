function res = fra_faces_baw_face_only(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    addpath('~/code/mircs');
    config;
    load ~/code/mircs/s40_fra_faces;
    res.fra_db = s40_fra_faces;
    res.conf = conf;
    cd /home/amirro/code/3rdparty/voc-release5
    startup
    load ~/code/3rdparty/dpm_baseline.mat
    res.model = model;
    return;
end

model = reqInfo.model;
conf_ = reqInfo.conf;
if (nargin == 4)
    fra_db = moreParams;
else
    fra_db = reqInfo.fra_db;
end

k = findImageIndex(fra_db,I);
curImageData = fra_db(k);

conf_.get_full_image = true;
[I,I_rect] = getImage(conf_,curImageData);
I_box = curImageData.faceBox;
I_box = round(inflatebbox(I_box,1.5,'both',false));
detections = struct('rot',{},'boxes',{});
I = cropper(I,round(I_box));

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
I_orig = imResample(I,resizeFactor);

for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');
    [ds, bs] = imgdetect(I, model,-10);
    top = nms(ds, 0.1);
    detections(iRot).rot = rots(iRot);
    if (isempty(top))
        error('so low....');
        detections(iRot).boxes = -inf(1,6);
        %detections(iRot).boxes = curImageData.faceBox;
    else
        ds = ds(top(1:min(10,length(top))),:);
        ds(:,1:4) = ds(:,1:4)/resizeFactor+repmat(I_box([1 2 1 2]),size(ds,1),1);
        detections(iRot).boxes = ds;
    end
end
res.detections = detections;

% detections = struct('rot',{},'boxes',{});
% I_orig = I;
% %rots = -20:10:20;
% rots = 0;
% for iRot = 1:length(rots)
%     I = imrotate(I_orig,rots(iRot),'bilinear','crop');
%     [ds, bs] = imgdetect(I, model,-2);
%     top = nms(ds, 0.1);
%     if (isempty(top))
%         boxes = -inf(1,5);
%     end
%     detections(iRot).rot = rots(iRot);
%     detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
% end
% res.detections = detections;