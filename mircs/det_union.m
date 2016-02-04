function dets = det_union(old_dets)
% simply the union of all detections....
dets = initClusters;
dets(1).isvalid = true;
locs = {};
for k=1:length(old_dets)
    locs{k} = old_dets(k).cluster_locs;
end
locs = cat(1,locs{:});

[~,is] = sort(locs(:,12),'descend');
locs = locs(is,:);
[a,b,c] = unique(locs(:,11),'first');
locs = locs(b,:); 

[~,is] = sort(locs(:,12),'descend'); % make sure the order is preserved.

dets.cluster_locs = locs(is,:);

end