function res = my_facial_landmarks_new(conf,I,reqInfo,pipelineStruct)
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


fra_struct = face_detection_to_fra_struct(conf,pipelineStruct.funs(1).outDir,imageID);

% R = j2m(pipelineStruct.funs(1).outDir,imageID);
% %R = j2m('~/storage/s40_faces_baw',imageID);
% pipelineStruct = struct('baseDir',baseDir,'funs',funs);
% clear inputData;
% checkIfNeeded = true;
% for t = 2%1:4
%     inputData.inputDir = baseDir;
%     suffix =[];testing = true;
%     run_all_2(inputData,funs(t).outDir,funs(t).fun,testing,suffix,'mcluster01',checkIfNeeded,pipelineStruct);
% end
% if (~exist(R,'file'))
%     error('face detection file doesn''t exist');
% end
% r = load(R);
% detections = r.res.detections;
% fra_struct = struct;
% fra_struct.imageID = imageID;
% rots = [detections.rot];
% fra_struct.faceBox = detections(rots==0).boxes(1,1:4);
if (all(isinf(fra_struct.faceBox)))
    res = [];
    return;
end
fra_struct.faceBox = fra_struct.faceBox+I_rect([1 2 1 2]);
conf.get_full_image = true;
roiParams.centerOnMouth = false;
roiParams.infScale = 1.5;
roiParams.absScale = 192;
[rois,roiBox,I] = get_rois_fra(conf,fra_struct,roiParams);
faceBox = rois(1).bbox;
bb = round(faceBox(1,1:4));
[res.kp_global,res.kp_local] = myFindFacialKeyPoints(conf,I,bb,predData.XX,...
    predData.kdtree,predData.curImgs,predData.ress,predData.ptsData,kpParams);
res.roiBox = roiBox;
res.roiParams = roiParams;