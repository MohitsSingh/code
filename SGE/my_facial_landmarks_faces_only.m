function res = my_facial_landmarks_faces_only(conf,I,reqInfo,moreParams)
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
    load ~/code/mircs/s40_fra_faces_d.mat;
    res.fra_db = s40_fra_faces_d;
    return;
end
conf = reqInfo.conf;
imageID = I;
fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,imageID);
[I_orig,I_rect] = getImage(conf,fra_db(k));
kpParams.debug_ = false;
kpParams.wSize = reqInfo.wSize;
kpParams.extractHogsHelper = reqInfo.extractHogsHelper;
kpParams.im_subset = reqInfo.predData.im_subset;
kpParams.requiredKeypoints = reqInfo.requiredKeypoints;
predData = reqInfo.predData;
% R = j2m('~/storage/s40_faces_baw',imageID);
% if (~exist(R,'file'))
%     error('face detection file doesn''t exist');
% end
% r = load(R);
% detections = r.res.detections;
% s40_fra = struct;
% s40_fra.imageID = imageID;
% s40_fra.faceBox = detections.boxes(1,1:4);
% if (all(isinf(s40_fra.faceBox)))
%     res.roiBox = [];
%     res.roiParams = [];
%     res.kp_global = [];
%     res.kp_local = [];
%     return;
% end
% s40_fra.faceBox = s40_fra.faceBox+I_rect([1 2 1 2]);
conf.get_full_image = true;
roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
[rois,roiBox,I,scaleFactor] = get_rois_fra(conf,fra_db(k),roiParams);
faceBox = rois(1).bbox;
bb = round(faceBox(1,1:4));
kpParams.debug_ = false;
try
    [res.kp_global,res.kp_local] = myFindFacialKeyPoints(conf,I,bb,predData.XX,...
        predData.kdtree,predData.curImgs,predData.ress,predData.ptsData,kpParams);
    
    
    res.kp_global(:,1:4) = transformToOriginalImageCoordinates(res.kp_global(:,1:4),scaleFactor,roiBox);
    res.kp_local(:,1:4) = transformToOriginalImageCoordinates(res.kp_local(:,1:4),scaleFactor,roiBox);
catch e
    res.kp_global = []
    res.kp_local = []
end
    
    % x2(I_orig); plotBoxes(res.kp_local);
    
    res.roiBox = roiBox;
    res.roiParams = roiParams;