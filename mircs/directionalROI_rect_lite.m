function xy = directionalROI_rect_lite(p0,p1,roiWidth)
if nargin == 1
    p1 = p0(:,3:4);
    roiWidth = p0(:,5);
    p0 = p0(:,1:2);
end
v1 = (p1-p0);
v2 = roiWidth*v1/norm(v1);
v2 = [-v2(2) v2(1)];
p_center = (p1+p0)/2;

isClockWise = det([v1 0;v2 0;1 1 1])>0;
if ~isClockWise
    v2 = -v2;
end
% det(A)

p3 = p_center+(v1+v2)/2;
p2 = p_center+(-v1+v2)/2;
p1 = p_center+(-v1-v2)/2;
p0 = p_center+(v1-v2)/2;

xy = [p0;p1;p2;p3];


% function xy = directionalROI_rect_lite(startPt,theta,roiWidth,roiLength,isCenter)
% if nargin < 4
%     roiWidth = 10;
% end
% if nargin < 4
%     roiLength = 2*roiWidth;
% end
% 
% if nargin < 5
%     isCenter = false;
% end
% 
% v1 = [sin(theta) cos(theta)];
% 
% if isCenter
%     startPt = startPt-v1*roiLength/2;
%     %xy = bsxfun(@minus,xy,);
% end
% 
% v2 = [-v1(2) v1(1)];
% p0 = startPt-roiWidth*v2/2;
% p1 = p0+v1*roiLength;
% p2 = p1+roiWidth*v2;
% p3 = p2-v1*roiLength;
% xy = [p3;p2;p1;p0];
% 
