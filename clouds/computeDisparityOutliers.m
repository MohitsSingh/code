function [norms,norm_diffs] = computeDisparityOutliers(xy_src,xy_dst)

vec_norms = @(x) sum(x.^2,2).^.5;
% for each point, find the distribution of its nearest neighbor's movement
% and remove it if it doen't make sense.
knn = 5;
norms = vec_norms(xy_src-xy_dst)
d1 = l2(xy_src,xy_src).^.5;
norm_diffs = zeros(size(norms));
for t = 1:size(xy_src,1)
    curPoint = xy_src(t,:);
    curNeighbors = find(d1(t,:)<knn);
    thisNorm = norms(t);
    neighborNorms = norms(curNeighbors);
    norm_diffs(t) = thisNorm-median(neighborNorms);
end

end

