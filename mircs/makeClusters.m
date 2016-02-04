function clusters = makeClusters(samples,locs)
clusters = initClusters;
for k = 1:size(samples,2)
    clusters(k).cluster_center = double(samples(:,k));
    clusters(k).cluster_samples= double(samples(:,k));
    if (~isempty(locs))
        clusters(k).cluster_locs= locs(k,:);
    end
    
    clusters(k).isvalid = true;
    clusters(k).w = samples(:,k);
    clusters(k).b = 0;
end
end