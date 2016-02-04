function [clusters,sel_] = removeInvalidClusters(clusters,min_k)
if (nargin < 2)
    min_k = 3;
end
for k = 1:length(clusters)
    clusters(k).isvalid = size(clusters(k).cluster_samples,2) >= min_k;
end
sel_ = [clusters.isvalid];
clusters(~[clusters.isvalid]) = [];