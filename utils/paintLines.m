function [Z,allPts] = paintLines(Z,segs,inds)
%PAINTLINES Summary of this function goes here
%   Detailed explanation goes here

% segs is for form x1,y1,x2,y2 (possibly repeated as rows).
if (size(segs,2) == 2) % this is a set of points, turn into segments
    segs = [segs(1:end-1,:) segs(2:end,:)];
end
allPts = {};
p = prod(size(Z));
if (nargin < 3)
    inds = 1:size(segs,1);
end

for k = 1:size(segs,1)
    rs = segs(k,2);
    cs = segs(k,1);
    re = segs(k,4);
    ce = segs(k,3);
    [yy,xx] = LineTwoPnts(rs,cs, re,ce); % y,x
    yy = yy(:);
    xx = xx(:);
    idx = sub2ind2(size2(Z),[yy(:),xx(:)]);
    goods = idx > 0 & idx <=p;
    idx = idx(goods);
    yy = yy(goods);
    xx = xx(goods);
    allPts{k} = [xx yy];
    %idx     = sub2ind(size(Z),yy,xx);
    
    %     goods
    % remove indices where already painted to avoid
    %elems = SegInMat(Z, rs, cs, re, ce);
    Z(idx) = inds(k);
end