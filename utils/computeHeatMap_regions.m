function [map] = computeHeatMap_regions(img,regions,scores,mode)
if (nargin < 3)
    mode = 'sum';
end
if (strcmp(mode,'max'))
    map = -inf(size(img,1),size(img,2));
else
    map = zeros(size(img,1),size(img,2));
end
counts = zeros(size(map));
sz = dsize(img,1:2);

for idx = 1:length(regions)
           
    score = scores(idx);
    if (strcmp(mode,'max'))
        maskBox = -inf(size(img,1), size(img,2));
    else
        maskBox = zeros(size(img,1), size(img,2));
    end
    curRegion = regions{idx};
    maskBox(curRegion) = score;
    if (strcmp(mode,'max'))
        map = max(map,maskBox);
    else
        map = map + maskBox;
    end
    
end
if (size(regions,1) && strcmp(mode,'max'))
    infs = isinf(map);
    map(infs) = min(map(~infs))-std(map(~infs))-eps;
end
end