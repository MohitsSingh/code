function res = fra_obj_pred_face_only(conf,I,reqInfo,moreParams)
if (nargin == 0)
    cd ~/code/mircs;
    initpath;
    config;
    res.conf = conf;
    addpath('~/code/SGE');
    addpath(genpath('~/code/utils'));
    load ~/code/mircs/s40_fra_faces_d.mat
    addpath('/home/amirro/code/mircs/');
    res.fra_db = s40_fra_faces_d;
    load ~/storage/misc/actionObjectPredData.mat;
    res.XX = XX;
    res.offsets = offsets;
    res.all_scales = all_scales;
    res.imgInds = imgInds;
    res.subInds = subInds;
    res.values = values;
    res.imgs = imgs;
    res.masks = masks;
    res.all_boxes = all_boxes;
    res.origImgInds = origImgInds;
    res.curParams= curParams;
    res.kdtree = vl_kdtreebuild(XX,'Distance','L2');
    res.debugParams = debugParams;
    return;
end

%% experiment 0043 -
%% 19/10/2014
XX = reqInfo.XX;
offsets = reqInfo.offsets;
all_scales = reqInfo.all_scales;
imgInds = reqInfo.imgInds;
subInds = reqInfo.subInds;
values = reqInfo.values;
imgs = reqInfo.imgs;
masks = reqInfo.masks;
curParams = reqInfo.curParams;
all_boxes = reqInfo.all_boxes;
origImgInds = reqInfo.origImgInds;
kdtree = reqInfo.kdtree;
debugParams = reqInfo.debugParams;

%%
close all
res.pMap = [];
res.shapeMask = [];
res.roiBox = [];
res.scaleFactor = [];
res.boxes = [];
res.boxes_orig = [];
debugParams.debug = false;
toOverride = false;
% resDir_orig = '~/storage/s40_fra_box_pred_2014_09_17';
% resDir = '~/storage/s40_fra_box_pred_full';
% ensuredir(resDir);
u = 0;
load s40_fra; fra_db = s40_fra;

fra_db = reqInfo.fra_db;
k = findImageIndex(fra_db,I);
curImageData = fra_db(k);
curImageData = switchToGroundTruth(curImageData);
isTrainImg = curImageData.isTrain & curImageData.indInFraDB~=-1;
if (isTrainImg)
    return
end
if (~curImageData.valid),return,end
%             if (fra_db(t).classID~=3),continue,end
u = u+1

% resPath = fullfile(resDir,[curImageData.imageID '_' curParams.objType '.mat']);
%
% if (toOverride || ~exist(resPath,'file') || debugParams.debug)
curParams.extent = 2;
curParams.img_h = 70*curParams.extent/1.5;
curParams.nn = 10; % was 10
curParams.stepSize = 3;
imgRots = -30:30:30;
bestRot = 0;
theRot = 0;
maxScore = -inf;
for flips = [0 1]
    pp = zeros(size(imgRots));
    for iRot = 1:length(imgRots)
        curParams.rot =  imgRots(iRot);
        curParams.flip = flips;
        debugParams.keepAllVotes = true;
        [pMap,I,roiBox,scaleFactor] = predictBoxes_fra(conf,curImageData,XX,curParams,offsets,all_scales,imgInds,subInds,values,imgs,masks,all_boxes,kdtree,origImgInds,debugParams);
        u = max(pMap(:));
        if (u > maxScore)
            %                             u
            maxScore = u;
            bestFlip = flips;
            bestRot = imgRots(iRot);
        end
    end
end

curParams.stepSize = 1;
curParams.rot = bestRot;
curParams.flip = bestFlip;
[pMap,I,roiBox,scaleFactor,shapeMask] = predictBoxes_fra(conf,curImageData,XX,curParams,offsets,all_scales,imgInds,subInds,values,imgs,masks,all_boxes,kdtree,origImgInds,debugParams);
if (curParams.flip)
    I = flip_image(I);
    pMap = flip_image(pMap);
    shapeMask = flip_image(shapeMask);
end
I = imrotate(I,-bestRot,'bilinear','crop');
pMap = imrotate(pMap,-bestRot,'bilinear','crop');
shapeMask = imrotate(shapeMask,-bestRot,'bilinear','crop');
if (debugParams.debug)
    figure(1);clf;
    vl_tightsubplot(2,2,2);
    imagesc2(sc(cat(3,pMap.^2,I),'prob_jet'));
    boxes = pMapToBoxes(pMap,20,3);plotBoxes(boxes);
    vl_tightsubplot(2,2,1);
    imagesc2(I);
    vl_tightsubplot(2,2,3); imagesc2(sc(cat(3,shapeMask.^2,I),'prob_jet'));
    II = getImage(conf,curImageData);
    boxes_orig = boxes/scaleFactor+repmat([roiBox([1 2 1 2]) 0],size(boxes,1),1);
    pause;drawnow
else
    boxes = pMapToBoxes(pMap,20,3);
    boxes_orig = boxes/scaleFactor+repmat([roiBox([1 2 1 2]) 0],size(boxes,1),1);
    res.pMap = pMap;
    res.shapeMask = shapeMask;
    res.roiBox = roiBox;
    res.scaleFactor = scaleFactor;
    res.boxes = boxes;
    res.boxes_orig = boxes_orig;
end

% end
