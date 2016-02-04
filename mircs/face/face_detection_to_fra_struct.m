function fra_struct = face_detection_to_fra_struct(conf,faceDir,imageID,kpPredDir)

% R = j2m(faceDir,imageID);
R = j2m('~/storage/s40_faces_baw',imageID);

if (~exist(R,'file'))
    error('face detection file doesn''t exist');
end
r = load(R);
if (isfield(r,'res'))
    r = r.res;
end
detections = r.detections;
fra_struct = struct;
fra_struct.imageID = imageID;
fra_struct.valid = true;
rots = [detections.rot];
if isempty(rots) || isempty(detections(rots==0).boxes)
    [fra_struct.faceBox,fra_struct.faceScore,fra_struct.faceComp,fra_struct.valid] = deal([],[],[],false);
    return
end

fra_struct.faceBox = detections(rots==0).boxes(1,1:4);
fra_struct.faceScore = detections(rots==0).boxes(1,end);
fra_struct.faceComp = detections(rots==0).boxes(1,5);

if (all(isinf(fra_struct.faceBox)))
    fra_struct.valid = false;
    return;
end

if (nargin < 4)
    return;
end
curOutPath = j2m(kpPredDir,fra_struct);
L = load(curOutPath);%,'curKP_global','curKP_local');
if (~isfield(L,'res'))
    res = L;
end

global_pred = res.kp_global;
local_pred = res.kp_local;
preds = local_pred;
bc1 = boxCenters(global_pred);
bc2 = boxCenters(local_pred);
bc_dist = sum((bc1-bc2).^2,2).^.5;
bad_local = bc_dist > 30;
goods_1= global_pred(:,end) > 2;
local_pred(bad_local,1:4) = global_pred(bad_local,1:4);
goods = goods_1 & ~bad_local;
kp_preds = local_pred;
goods = true(size(kp_preds,1),1);
kp_centers = boxCenters(kp_preds);
nKP = size(kp_centers,1);
fra_struct.keypoints.kp_centers = kp_centers;
fra_struct.keypoints.goods = goods;
fra_struct.keypoints.global_pred = global_pred;
fra_struct.keypoints.local_pred = local_pred;


roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
% [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_struct,roiParams);

%fra_struct.mouth = boxCenters(kp_preds(3,:))/scaleFactor+roiBox(1:2);
fra_struct.mouth = boxCenters(kp_preds(3,:));%/scaleFactor+roiBox(1:2);

fra_struct.raw_faceDetections = detections(rots==0);