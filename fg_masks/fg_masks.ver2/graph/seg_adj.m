function [segs] = seg_adj(map)
%SEG_ADJ Summary of this function goes here
%   Detailed explanation goes here
map1 = circshift(map, [1 0]);
map1(1,:) = map(1,:);
map2 = circshift(map, [-1 0]);
map2(end,:) = map(end,:);
map3 = circshift(map, [0 1]);
map3(:,1) = map(:,1);
map4 = circshift(map, [0 -1]);
map4(:,end) = map(:,end);
labels = unique(map(:));
rprops = regionprops(map,'PixelIdxList');
for i=1:length(labels)
    ind = rprops(i).PixelIdxList;%find(map(:) == labels(i));
    % %     if (any(setdiff(rprops(i).PixelIdxList,ind)))
    %         error('aha')
    %     end    
    segs(i).ind = ind;
    segs(i).count = length(ind);
    adj = [map1(ind) map2(ind) map3(ind) map4(ind)];
    adj = unique(adj(:));
    adj = setdiff(adj, labels(i));
    segs(i).adj = adj;
end

end

