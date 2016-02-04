function res = fra_selective_search_feats(conf,I,reqInfo)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV'));
    return;
end

%%
%S
[learnParams,conf] = getDefaultLearningParams(conf,1024);
featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];

roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.useCenterSquare = false;
[mcg_boxes,I] = get_mcg_boxes(conf,curImageData,roiParams);
feats = {};
rois = {};
for iRoi = 1:size(mcg_boxes,1);
    rois{end+1} = poly2mask2(mcg_boxes(iRoi,:),size2(I));
end
res = featureExtractor.extractFeatures(I,rois,'normalization','Improved');