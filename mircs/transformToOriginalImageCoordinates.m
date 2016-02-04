function b = transformToOriginalImageCoordinates(bb,scaleFactor,roiBox)
b = (bb/scaleFactor)+repmat(roiBox(1:2),size(bb,1),size(bb,2)/2);
end