function res = fra_landmarks_parallel(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    addpath('/home/amirro/code/3rdparty/face-release1.0-basic/'); % zhu + ramanan facial landmarks
    config;
    %% 1. Parameter Settings
    res.doFrameRemoving = true;
    res.useSP = true;
    res.conf = conf;
    %     load fra_db;
    %     res.fra_db = fra_db;
%     load s40_fra;
    load ~/code/mircs/s40_fra_faces_d
    res.fra_db = s40_fra_faces_d;
    return;
end

if (any(strfind(I,'aflw_cropped_context')))
    I = imread(I); % run the zhu+ramanan code.
    res.landmarks = extractLandmarks(conf,I,-20:20:20,{'face_p146_small'});
else
    fra_db = reqInfo.fra_db;
    k = findImageIndex(fra_db,I);
    curImageData = fra_db(k);
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    roiParams.centerOnMouth = false;    
    curImageData = switchToGroundTruth(curImageData);    
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImageData,roiParams);
    landmarks = extractLandmarks(conf,I);
    res.roiBox = roiBox;
    res.landmarks = landmarks;
    res.roiParams = roiParams;
end