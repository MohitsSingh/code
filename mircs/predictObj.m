function [ res ] = predictObj(conf,curImageData,initData)
%PREDICTOBJ Summary of this function goes here
%   Detailed explanation goes here
curParams = initData.curParams;
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
        initData.debugParams.keepAllVotes = true;
        [pMap,I,roiBox,scaleFactor] = predictBoxes_fra(conf,curImageData,initData.XX,...
            curParams,initData.offsets,initData.all_scales,initData.imgInds,initData.subInds,initData.values,initData.imgs,...
            initData.masks,initData.all_boxes,initData.kdtree,initData.origImgInds,initData.debugParams);
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



[pMap,I,roiBox,scaleFactor,shapeMask] = predictBoxes_fra(conf,curImageData,initData.XX,curParams,initData.offsets,...
    initData.all_scales,initData.imgInds,initData.subInds,initData.values,initData.imgs,initData.masks,...
    initData.all_boxes,initData.kdtree,initData.origImgInds,initData.debugParams);
if (curParams.flip)
    I = flip_image(I);
    pMap = flip_image(pMap);
    shapeMask = flip_image(shapeMask);
end
I = imrotate(I,-bestRot,'bilinear','crop');
pMap = imrotate(pMap,-bestRot,'bilinear','crop');
shapeMask = imrotate(shapeMask,-bestRot,'bilinear','crop');
if (initData.debugParams.debug)
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

end

