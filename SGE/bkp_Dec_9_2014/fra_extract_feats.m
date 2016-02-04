function res = fra_extract_feats(conf,I,reqInfo,forceTestMode)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    return;
end
fra_db = reqInfo.fra_db;
[learnParams,conf] = getDefaultLearningParams(conf,1024);
% make sure class names corrsepond to labels....
roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;

featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
%%

%S
% res = struct('feat',{},'type',{},'name',{},'isTrain',{},'srcImageIndex',{},...
%     'is_gt_location',{},'flipped',{},'isValid',{});
clear res;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.useCenterSquare = false;
res.imgFeats = fra_extract_feats_helper(conf,curImageData,roiParams,featureExtractor);
roiParams.useCenterSquare = true;
res.imgFeatsSquare = fra_extract_feats_helper(conf,curImageData,roiParams,featureExtractor);

