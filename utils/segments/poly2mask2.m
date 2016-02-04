function [ bw ] = poly2mask2(pts,sz)
%POLY2MASK2 convenient version of poly2mask
%   Pts is an array of Nx2 (X,Y) pairs. s1 may either be [rows cols] or
% if s2 is given, s1 is regarded as rows and s2 as cols.
% if pts is a 1x4 array, it is transformed to a polygon.

if (iscell(pts))
    bw = poly2mask2(pts{1},sz);
    for t = 2:length(pts)
        bw = bw | poly2mask2(pts{t},sz);
    end
    return;
end
if numel(sz)>3
    sz = size2(sz);
end
if (all(size(pts)==[1 4]))
    pts = box2Pts(pts);
end

bw = poly2mask(pts(:,1),pts(:,2),sz(1),sz(2));

end