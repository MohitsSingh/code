function [ detections ] = anchoredBoxSamples(I,curMouthPts,myParam)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
myParam.objToFaceRatio = .2;
wndSize = round(size(I,1)*myParam.objToFaceRatio);
xx = col(linspace(curMouthPts(1,1),curMouthPts(2,1),myParam.ptsAlongLine)-wndSize/2);
yy = ones(size(xx))*mean(curMouthPts(:,2));
detections = zeros(length(xx),5);
detections(:,1:4) = round([xx yy,xx+wndSize,yy+wndSize]);

end

