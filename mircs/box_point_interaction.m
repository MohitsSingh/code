function res = box_point_interaction(boxes,I,mouthCenter)
% interaction features for boxes and some point
s = size(I,1);
b1 = boxes/s;
[corners,inds] = boxes2Polygons(boxes);
mouthCenter1 = repmat(mouthCenter,4,1);
corner_diffs = col(cellfun2(@(x) x-mouthCenter1,corners));
corner_dists = cellfun2(@(x) sum(x.^2,2).^.5,corner_diffs);
b2 = cellfun(@min,corner_dists)/s;
b3 = cellfun(@max,corner_dists)/s;
[nRows,nCols,a] = BoxSize(boxes);
b4 = a/prod(size2(I)); % ratio to image area
b5 = [nRows nCols]/s; % width / height in image
b6 = cellfun2(@(x) reshape(x,1,[]),corner_diffs);
b6 = cat(1,b6{:})/s;
% corners w.r.t mouth
res = ([b1 b2 b3 b4 b5 b6])';
res = vl_homkermap(res,1);
end