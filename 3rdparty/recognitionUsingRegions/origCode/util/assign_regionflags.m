function regionflags = assign_regionflags(regions, bboxes, frac)
% function regionflags = assign_regionflags(regions, bboxes, frac)
%
% This function assigns binary labels to regions based on whether the
% region "lies in" a rectangular area (specified by a bounding box). We
% claim the region is "in" if no less than a certain fracion of the region
% area is actually inside the bounding box.
%
% Copyright @ Chunhui Gu, April 2009

if nargin < 3,
    frac = 0.8;
end;

mask = false(size(regions{1}));
for bb = 1:size(bboxes,1),
    mask(bboxes(bb,2):bboxes(bb,4),bboxes(bb,1):bboxes(bb,3)) = true;
end;

regionflags = false(length(regions),1);
for rId = 1:length(regions),
    int = regions{rId} & mask;
    if sum(int(:)) / sum(regions{rId}(:)) > frac,
        regionflags(rId) = true;
    end;
end;