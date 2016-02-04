function [roi,roi_out] = getActionRoi(I,landmarks,model) %assume mouth is in the middle.
if (nargin < 3)
    actionRoiAngle = 50;
else
    actionRoiAngle = model.actionRoiAngle;
end
posemap = 90:-15:-90;
curPose = posemap(landmarks.c);
if (isfield(model,'lookDirection'))
    curPose = sign(curPose)*model.lookDirection;
end
pp = curPose;
vec = [sind(pp),cosd(pp)];
h = size(I,1);
roi_center = [size(I,2),size(I,1)]/2;
roi = directionalROI(I,roi_center,vec',actionRoiAngle);
roi_out = directionalROI(I,roi_center,vec',actionRoiAngle+20);