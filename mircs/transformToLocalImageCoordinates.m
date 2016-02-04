function b = transformToLocalImageCoordinates(bb,scaleFactor,roiBox)
b = (bb-repmat(roiBox(1:2),size(bb,1),size(bb,2)/2))*scaleFactor;
end