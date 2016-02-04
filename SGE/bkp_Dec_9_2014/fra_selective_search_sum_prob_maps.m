function res = fra_selective_search_sum_prob_maps(conf,I,reqInfo)
if (nargin == 0)
    clear res;
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    res.conf = conf;
    load fra_db.mat;
    res.fra_db = fra_db;
    R = load('~/storage/misc/baseLine_classifiers_5_regular_Improved64_try.mat');
    classifiers  = [R.res.classifier];
    ws = cat(2,classifiers.w);
    res.ws = ws(1:end-1,:);
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV'));
    return;
end

%%
%S
ws = reqInfo.ws';

roiParams.infScale = 3.5;
roiParams.absScale = 200*roiParams.infScale/2.5;
fra_db = reqInfo.fra_db;
all_class_names = {fra_db.class};
class_labels = [fra_db.classID];
classes = unique(class_labels);
% make sure class names corrsepond to labels....
[lia,lib] = ismember(classes,class_labels);
classNames = all_class_names(lib);
roiParams.useCenterSquare = false;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
roiParams.useCenterSquare = false;
[mcg_boxes,I] = get_mcg_boxes(conf,curImageData,roiParams);
objTypes = {'head','hand','obj','mouth'};
areas = sum_boxes(ones(size2(I)),mcg_boxes);
sums = -inf(size(mcg_boxes,1),length(classes),length(objTypes));
for iClass = 1:length(classes)
    for iObjType = 1:length(objTypes)
        
        probPath = fullfile('~/storage/s40_fra_box_pred_new',[curImageData.imageID '_' classNames{iClass} '_' objTypes{iObjType} '.mat']);
        load(probPath);
        pMap = imResample(pMap,size2(I))/100;
        sums(:,iClass,iObjType) = sum_boxes(pMap,mcg_boxes);
    end
end

res.areas = areas;
res.sums = sums;
