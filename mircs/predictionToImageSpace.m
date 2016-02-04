function [pMap,objMap] = predictionToImageSpace(L,I_orig,roiBox)
boxFrom = L.roiBox;
boxTo = [1 1 fliplr(size2(L.pMap))];

%     S1 = foregroundSaliency(conf,curImageData.imageID);
%     S1 = cropper(S1,roiBox);
T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
pMap = imtransform(L.pMap,T,'XData',[1 size(I_orig,2)],'YData',[1 size(I_orig,1)],'XYScale',1);
objMap = imtransform(L.shapeMask,T,'XData',[1 size(I_orig,2)],'YData',[1 size(I_orig,1)],'XYScale',1);
if (nargin == 3)
    pMap = cropper(pMap,roiBox);
    objMap = cropper(objMap,roiBox);
end
end