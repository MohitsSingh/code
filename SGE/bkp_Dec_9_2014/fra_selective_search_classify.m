function res = fra_selective_search_classify(conf,I,reqInfo)
if (nargin == 0)
    clear res;
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    R = load('~/storage/misc/baseLine_classifiers_5_regular_Improved64_try1x2.mat');
    classifiers  = [R.res.classifier];
    ws = cat(2,classifiers.w);
    res.ws = ws(1:end-1,:);
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV'));
    return;
end

%%
%S
[learnParams,conf] = getDefaultLearningParams(conf,1024);
featureExtractor = learnParams.featureExtractors{1}; % check this is indeed fisher
featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
ws = reqInfo.ws';
roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.useCenterSquare = false;
[mcg_boxes,I] = get_mcg_boxes(conf,curImageData,roiParams);
% mcg_boxes = mcg_boxes(1:5,:);
rois = {};
for iRoi = 1:size(mcg_boxes,1);
    rois{end+1} = poly2mask2(mcg_boxes(iRoi,:),size2(I));
end
imgFeats = featureExtractor.extractFeatures(I,rois,'normalization','Improved');
res.scores = ws*imgFeats;
clear imgFeats;
% now flip...
I = flip_image(I);
rois = cellfun2(@flip_image,rois);
imgFeats = featureExtractor.extractFeatures(I,rois,'normalization','Improved');
res.scores_flip = ws*imgFeats;
