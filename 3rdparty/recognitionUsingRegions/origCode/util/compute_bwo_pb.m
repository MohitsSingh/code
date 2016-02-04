function f = compute_bwo_pb(pb_I, region, width, height)
% function f = compute_bwo_pb(pb_I, region, width, height)
%
% This function divides the bounding box of the input region into grids,
% and collects gPb boundary responses in each cell of the grid.
%
% Copyright @ Chunhui Gu, April 2009

mask = (region > 0);

[imy,imx,nbins] = size(pb_I);
[X,Y] = meshgrid(1:imx,1:imy);

X = X(mask); Y = Y(mask);
X = (X-1)/(size(pb_I,2)-1);
Y = (Y-1)/(size(pb_I,1)-1);

xi = max(1,min(width,round(X*width+0.5)));
yi = max(1,min(height,round(Y*height+0.5)));
sbi = (xi-1)*height+yi;

m = zeros(sum(mask(:)),nbins);
for i = 1:nbins,
    tmp = pb_I(:,:,i);
    m(:,i) = tmp(mask);
end;
tmp = repmat(max(m, [], 2), [1 nbins]);
m = (m == tmp) .* m;

f = zeros(width*height, nbins);
for j = 1:length(sbi),
    f(sbi(j),:) = f(sbi(j),:) + m(j,:);
end;

f(isnan(f)) = 0;
f = f / sum(sum(f));