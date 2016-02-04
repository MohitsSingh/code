function [ bw ] = poly2mask2_old(pts,s1,s2)
%POLY2MASK2 convenient version of poly2mask
%   Pts is an array of Nx2 (X,Y) pairs. s1 may either be [rows cols] or
% if s2 is given, s1 is regarded as rows and s2 as cols.
% if pts is a 1x4 array, it is transformed to a polygon.

if (all(size(pts)==[1 4]))
    pts = box2Pts(pts);
end
if (nargin < 3)
    s2 = s1(2);
    s1 = s1(1);
end

bw = poly2mask(pts(:,1),pts(:,2),s1,s2);

end