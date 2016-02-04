function regions = fillRegionGaps(regions,b)
if (nargin == 2 && b)
    regions = cellfun2(@(x) imdilate(x,ones(3)),regions);
else
    regions = cellfun2(@(x) imclose(x,ones(3)),regions);
end
end