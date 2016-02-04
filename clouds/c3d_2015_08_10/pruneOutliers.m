function [xy_src,xy_dst,z] = pruneOutliers(xy_src,xy_dst,fMatrix,mode)
if nargin < 3
    mode = 0;
end
if mode==0
    z = 0;
    [fMatrix, epipolarInliers, status] = estimateFundamentalMatrix(...
        xy_src, xy_dst, 'Method', 'RANSAC', ...
        'NumTrials', 1000, 'DistanceThreshold', .5, 'Confidence', 99.99);
    xy_src = xy_src(epipolarInliers, :);
    xy_dst = xy_dst(epipolarInliers, :);
else
    x1 = [xy_src ones(size(xy_src,1),1)];
    x2 = [xy_dst ones(size(xy_dst,1),1)]
    n = size(xy_src,1);
    z = zeros(n,1);
    for p = 1:size(xy_src,1)
        z(p) = x1(p,:)*fMatrix*x2(p,:)';
    end
    epipolarInliers = abs(z) <.3;
    xy_src = xy_src(epipolarInliers, :);
    xy_dst = xy_dst(epipolarInliers, :);
end
end