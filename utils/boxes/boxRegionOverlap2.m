function [ovp,ints] = boxRegionOverlap2(bbox,regions)
bbox = round(bbox);
ints = cellfun3(@(x) nnz(x(bbox(2):bbox(4),bbox(1):bbox(3))),regions);
region_areas = cellfun3(@(x) nnz(x),regions);

regions_not_boxes = region_areas-ints;
[~,~,a] = BoxSize(bbox);

ovp = ints./(a+regions_not_boxes);

end


