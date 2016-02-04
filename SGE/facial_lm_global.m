function res = facial_lm_global(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs
    initpath
    config
    conf.get_full_image = true;
    roiParams = defaultROIParams();    
    landmarkParams = load('~/storage/misc/kp_pred_data.mat');
    ptNames = landmarkParams.ptsData;
    ptNames = {ptNames.pointNames};
    requiredKeypoints = unique(cat(1,ptNames{:}));
    landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX,'Distance','L2');
    landmarkParams.conf = conf;
    landmarkParams.wSize = 96;
    landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
    landmarkParams.requiredKeypoints = requiredKeypoints;
    %{'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    landmarkInit = landmarkParams;
    landmarkInit.debug_ = false;%%
    %     conf.get_full_image = true;
    res.conf = conf;
    res.landmarkInit = landmarkInit;
    load ~/storage/misc/s40_fra_faces_d_new
    res.fra_db = s40_fra_faces_d;
    load ~/storage/misc/s40_face_detections.mat; % all_detections
    res.all_detections = all_detections;
    return;
end

conf = initData.conf;
conf.get_full_image = false;
landmarkInit = initData.landmarkInit;
fra_db = initData.fra_db;

all_detections = initData.all_detections;
k = findImageIndex(fra_db,params.name);
imgData = fra_db(k);
[I_orig,I_rect] = getImage(conf,imgData);
curFaceBox = all_detections(k).detections.boxes(1,:);
res.kp_global = zeros(length(landmarkInit.requiredKeypoints),5);
if (isnan(curFaceBox(1)) || isinf(curFaceBox(1)))
    return;
end

curScore = curFaceBox(end);
curFaceBox = round(curFaceBox(1:4));
I = cropper(I_orig,curFaceBox);
I_crop_orig = I;
resizeFactor = 128/size(I,1);
I = imResample(I,[128 128],'bilinear');
bb = [1 1 fliplr(size2(I))];
[kp_global] = myFindFacialKeyPoints(conf,I,bb,landmarkInit.XX,...
    landmarkInit.kdtree,landmarkInit.curImgs,landmarkInit.ress,landmarkInit.ptsData,landmarkInit);
kp_global(:,1:4) = kp_global(:,1:4)/resizeFactor;
kp_global(:,1:4) = bsxfun(@plus,kp_global(:,1:4),I_rect([1 2 1 2])+curFaceBox([1 2 1 2])-1);
res.kp_global = kp_global;

