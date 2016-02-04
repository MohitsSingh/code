function [hist, region_data] = compute_hists_new(pb_im, regions)
% function [hist, region_data] = compute_hists(imgFile, regions)
%
% The function computes HOG-based histograms from regions of the image. It
% also returns geometric features of the regions.
% 
% Related functions: compute_bwo_pb
%
% Copyright @ Chunhui Gu, April 2009

width = 4;
height = 4;

% load([imgFile '_gPb.mat'], 'gPb_orient');
% pb_im = gPb_orient;
% clear gPb_orient;

hist = zeros(length(regions),128);
region_data = zeros(length(regions),5);
for i = 1:length(regions),

    region = regions{i};
    [y,x] = find(region);

    new_pb_im = pb_im(min(y):max(y), min(x):max(x), :);
    new_region = region(min(y):max(y), min(x):max(x));
    feat = compute_bwo_pb(new_pb_im, new_region, width, height);
    hist(i,:) = reshape(feat, 1, []);

    region_data(i,:) = [size(new_pb_im, 1), ...
                        size(new_pb_im, 2), ...
                        sum(sum(new_region)), ...
                        size(pb_im, 1), ...
                        size(pb_im, 2)];
end;