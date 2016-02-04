function [ res ] = calc_flow( res, knn)
%CALC_FLOW Summary of this function goes here
%   Detailed explanation goes here
if nargin < 2
    knn = 10;
end
for t = 1:length(res)-1
    cur_xyz = res(t).xyz';
    next_xyz = res(t+1).xyz';
    tree = vl_kdtreebuild(next_xyz);
    [inds,dists] = vl_kdtreequery(tree,next_xyz,cur_xyz,'NumNeighbors',knn);
    
    M = zeros(size(cur_xyz));
    for tt = 1:size(M,2)
        M(:,tt) = mean(next_xyz(:,inds(:,tt)),2);
    end
    res(t).uvw = (M-cur_xyz)';
end

end

