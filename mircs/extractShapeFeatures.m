function [rprops,x] = extractShapeFeatures(regions)
rprops = {};
for k = 1:length(regions)
    
    
%     curRegion = cropToNonzero(regions{k});    
    rprops{k} = regionprops(regions{k},'Centroid','Area','Eccentricity','Solidity','MajorAxisLength','MinorAxisLength',...
        'Orientation');
    rprops{k} = rprops{k}(1);
end
n = numel(regions{1});
rprops = cat(1,rprops{:});
area_ = [rprops.Area];
eccen_ = [rprops.Eccentricity];
solid_ = [rprops.Solidity];
maj_ = [rprops.MajorAxisLength];
min_ = [rprops.MinorAxisLength];
ori_ = [rprops.Orientation];
centroid_ = cat(1,rprops.Centroid);

x = [centroid_';area_;eccen_;maj_;min_;solid_;cosd(ori_);sind(ori_)];
end