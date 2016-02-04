function [Z,Z_count] = paintRegionProps(labels,rprops,property,paintOnLabels)
Z = zeros(size(labels));
if (nargin == 4 && paintOnLabels)
    Z = labels;
end
if (iscell(rprops))
    regions = rprops;
    rprops = struct('PixelIdxList',{});
    for k = 1:length(regions)
        rprops(k).PixelIdxList = find(regions{k});
    end
end
if (isempty(rprops))
    rprops = regionprops(labels,'PixelIdxList');
end
% Z_count = zeros(size(Z));
for k = 1:length(rprops)
    k
    Z(rprops(k).PixelIdxList) = max(Z(rprops(k).PixelIdxList),property(k));
%     imagesc(Z);
%     pause;
%     Z(rprops(k).PixelIdxList) = plus(Z(rprops(k).PixelIdxList),property(k));
%     Z_count(rprops(k).PixelIdxList) = Z_count(rprops(k).PixelIdxList)+1;
    %     pause;
    %eval(['Z(rprops(' num2str(k) ').PixelIdxList) = rprops(' num2str(k) ').' property ';']);
end