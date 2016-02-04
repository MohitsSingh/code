function res = my_facial_landmarks(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    res.wSize = 96;
    res.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[res.wSize res.wSize],'bilinear')))) , y);
    res.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    res.predData = load('~/storage/misc/kp_pred_data.mat');
    res.predData.kdtree = vl_kdtreebuild(res.predData.XX);
    return;
end
conf = reqInfo.conf;
imageID = I;
[I_orig,I_rect] = getImage(conf,imageID);
kpParams.debug_ = false;
kpParams.wSize = reqInfo.wSize;
kpParams.extractHogsHelper = reqInfo.extractHogsHelper;
kpParams.im_subset = reqInfo.predData.im_subset;
kpParams.requiredKeypoints = reqInfo.requiredKeypoints;
predData = reqInfo.predData;
R = j2m('~/storage/s40_faces_baw',imageID);
if (~exist(R,'file'))
    error('face detection file doesn''t exist');
end
r = load(R);
detections = r.res.detections;
s40_fra = struct;
s40_fra.imageID = imageID;
s40_fra.faceBox = detections.boxes(1,1:4);
if (all(isinf(s40_fra.faceBox)))
    res.roiBox = [];
    res.roiParams = [];
    res.kp_global = [];
    res.kp_local = [];
    return;
end
s40_fra.faceBox = s40_fra.faceBox+I_rect([1 2 1 2]);
conf.get_full_image = true;
roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
[rois,roiBox,I,scaleFactor] = get_rois_fra(conf,s40_fra,roiParams);
faceBox = rois(1).bbox;
bb = round(faceBox(1,1:4));
kpParams.debug_ = false;

[res.kp_global,res.kp_local] = myFindFacialKeyPoints(conf,I,bb,predData.XX,...
    predData.kdtree,predData.curImgs,predData.ress,predData.ptsData,kpParams);

res.kp_global(:,1:4) = transformToOriginalImageCoordinates(res.kp_global(:,1:4),scaleFactor,roiBox);
res.kp_local(:,1:4) = transformToOriginalImageCoordinates(res.kp_local(:,1:4),scaleFactor,roiBox);

% x2(I_orig); plotBoxes(res.kp_local);

res.roiBox = roiBox;
res.roiParams = roiParams;