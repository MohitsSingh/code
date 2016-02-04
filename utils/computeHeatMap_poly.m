function [map] = computeHeatMap_poly(img,polys,scores,mode)
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

for idx = 1:length(polys)
           
    score = scores(idx);
    if (strcmp(mode,'max'))
        maskBox = -inf(size(img,1), size(img,2));
    else
        maskBox = zeros(size(img,1), size(img,2));
    end
    xy = polys{idx};
    maskBox(poly2mask2(double(xy),sz)) = score;
    if (strcmp(mode,'max'))
        map = max(map,maskBox);
    else
        map = map + maskBox;
    end
    
end
if (size(polys,1) && strcmp(mode,'max'))
    infs = isinf(map);
    map(infs) = min(map(~infs))-std(map(~infs));
end
end