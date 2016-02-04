function [regions] = processRegions(I_sub,candidates,mouthMask)
regions = candidates.masks;
mouth_box = region2Box(mouthMask);
[overlaps,ints] = boxesOverlap(candidates.bboxes,mouth_box);
regions(overlaps == 0) = [];
if isempty(regions)
    return;
end
overlaps = regionsOverlap3(regions,{mouthMask});
regions(overlaps == 0) = [];
regions = row(ezRemove(regions,I_sub,50,.3));

end