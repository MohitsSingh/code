function [ovp,ints,areas] = boxRegionOverlap(bbox,regions,sz,regionBoxes)
if (~iscell(regions)) % just a single region.
    regions = {regions};
end
if (nargin < 3 || isempty(sz))
    sz = size(regions{1});
end
if (numel(bbox)~=4) % assume then it is a mask
    Z = bbox;
    [yy,xx] = find(Z);
    bbox = pts2Box([xx yy]);
else % given as rectangle.
    Z = zeros(sz);
    Z = poly2mask2(box2Pts(bbox),sz);
    %Z = drawBoxes(Z,bbox,1,1);
    if (iscell(Z))
        Z = Z{1}>0;
    end
end

if (nargin == 4) % have region bounding boxes, so compute this first.
    sel_ = boxesOverlap(bbox,regionBoxes) > 0;
else
    sel_ = 1:length(regions);
end

Z = Z(:);
% regionBoxes = cellfun(@(x) pts2Box(ind2sub2(size(regions{1}),find(x))),regions,'UniformOutput',false);
ovp = zeros(size(regions));
ints = zeros(size(regions));
ints(sel_) = cellfun(@(x) sum(x(:) & Z(:)),regions(sel_));
if (nargout== 3)
    areas = cellfun(@(x) sum(x(:)), regions);
end
% uns = zeros(size(sel_));
uns = cellfun(@(x) sum(x(:) | Z),regions(sel_));
ovp(sel_) = ints(sel_)./uns;


end


