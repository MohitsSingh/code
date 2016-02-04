function [Z] = fillBoxes(sz,boxes,values,fillMode)
%FILLBOXES Summary of this function goes here
%   Detailed explanation goes here
Z = zeros(sz);
% fillMode: 1->max 2->assign 3->per pixel mean
if (nargin < 4)
    fillMode= 1;
end

for q = 1:length(values)
    %                 k = window_inds(q);
    b = round(boxes(q,1:4));
    b(1) = max(b(1),1);
    b(2) = max(b(2),1);
    b(3) = min(b(3),size(Z,2));
    b(4) = min(b(4),size(Z,1));
    if(fillMode==1)
        Z(b(2):b(4),b(1):b(3)) = max(Z(b(2):b(4),b(1):b(3)),values(q));
    elseif (fillMode == 2)
        Z(b(2):b(4),b(1):b(3)) = values(q);
    end
end


