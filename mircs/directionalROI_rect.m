function [res] = directionalROI_rect(startPt,theta,roiWidth,roiLength)
if nargin < 4
    roiWidth = 10;
end
if nargin < 4
    roiLength = 2*roiWidth;
end

v1 = [sind(theta) cosd(theta)];
v2 = [-v1(2) v1(1)];
p0 = startPt-roiWidth*v2/2;
p1 = p0+v1*roiLength;
p2 = p1+roiWidth*v2;
p3 = p2-v1*roiLength;
xy = [p0;p1;p2;p3];
startPoint = (p0+p3)/2;
endPoint = (p1+p2)/2;

res = struct('xy',xy,'startPoint',startPoint,'endPoint',endPoint,'theta',theta,'ispoly',true);