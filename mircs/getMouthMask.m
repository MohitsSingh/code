function [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,isTraining)
% function [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,isTraining)

if isTraining
    curLandmarks = [imgData.landmarks_gt.xy imgData.landmarks.goods(:)];
else
    curLandmarks = boxCenters(imgData.landmarks.xy);
end
windowCenter = round(fliplr(size2(I_sub))/2);
z = zeros(size2(I_sub));
z(windowCenter(2),windowCenter(1)) = 1;
z = bwdist(z) < mean(size2(I_sub))/8;
mouthMask = z;