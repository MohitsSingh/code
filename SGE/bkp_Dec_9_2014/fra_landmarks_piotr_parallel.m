function res = fra_landmarks_piotr_parallel(conf,I,reqInfo)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;    
    config;
%     addpath('~/code/utils');
    
    load ~/code/mircs/fra_db;
    res.fra_db = fra_db;
    res.conf = conf;
    cd /home/amirro/code/3rdparty/rcpr_v1
    load('data/rcpr.mat','regModel','regPrm','prunePrm');
    
    testFile=['data/COFW_test.mat'];
    load(testFile,'phisT','bboxesT');
    res.bboxesT=round(bboxesT);    
    res.regModel = regModel;
    res.regPrm = regPrm;
    res.prunePrm = prunePrm;
    
    return;
end
conf_ = reqInfo.conf;
fra_db = reqInfo.fra_db;

k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.infScale = 1.5;
roiParams.absScale = 100;
[rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf_,curImageData,roiParams);
% res.roiBox = roiBox;
% res.roiParams = roiParams;

[res] = detect_on_set({I},reqInfo.regModel,reqInfo.regPrm,reqInfo.prunePrm,[]);
res = res{1};
