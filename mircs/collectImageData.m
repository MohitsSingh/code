function [I,I_sub,imageData,hasObj] = collectImageData(conf,imgData,params)

% objMask,landmarks,landmarks_local
I = getImage(conf,imgData);
hasObj = false;
% 1. facial landmarks (e.g, around mouth);
[I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData);
imageData.faceBox = faceBox;
imageData.mouthBox = mouthBox;
dlib_landmark_split;
imageData.landmarks = imgData.Landmarks_dlib;
%mouthMask = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib);
[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
imageData.mouthMask = mouthMask;
imageData.landmarks_local = curLandmarks;
imageData.faceOutline = getFaceOutline(curLandmarks);
imageData.faceMask = poly2mask2(imageData.faceOutline,I_sub);    
gt_graph = get_gt_graph(imgData,params.nodes,params,I);
% 2. remove "trivial" regions
if ~isfield(gt_graph{2},'roiMask')
    return;
end
m = gt_graph{2}.roiMask;
if isempty(m) || nnz(m)==0
    return;
end
roiMask = cropper(m,mouthBox);
if (nnz(roiMask)==0)
    return;
end

hasObj = true;
imageData.roiMask = roiMask;