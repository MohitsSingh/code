function min_map = dist_bwo(hist1, hist2, region_data1, region_data2, dist_mode, aspect_con, scale_con)
% function min_map = dist_bwo(hist1, hist2, region_data1, region_data2, dist_mode, aspect_con, scale_con)
%
% returns the distance matrix
% hist1: blocked window histogram of image 1
% hist2: blocked window histogram of image 2
% region_data1, 2: width, height, pixel area information of each region
% aspect_con = flag put aspect ratio constraints on the comparison
%
% min_map(i, j) = distance from image1's histure #i to image2's feature #j
%
% Copyright @ Chunhui Gu, April 2009

if (nargin < 5)
    dist_mode = 'chi-square';
end;
if (nargin < 6)
    aspect_con = false;
    scale_con = false;
end

% Assuming histograms have been normalized as inputs
nhist1 = size(hist1,1);
nhist2 = size(hist2,1);

min_map = zeros(nhist1,nhist2,'single');
for j = 1:nhist1,
    temp = repmat(hist1(j,:),[nhist2 1]);
    switch dist_mode
        case 'chi-square'
            d = 0.5*sum((temp - hist2).^2 ./ ((temp+hist2)+((temp+hist2)==0)), 2);
        case 'l2'
            d = sqrt(sum((temp - hist2).^2,2));
    end;
    min_map(j,:) = d';
end;

if (aspect_con)
    th = 0.6;
    tmp1 = repmat(region_data1(:,2)./region_data1(:,1), [1 size(region_data2, 1)]);
    tmp2 = repmat((region_data2(:,2)./region_data2(:,1))', [size(region_data1, 1) 1]);
    mask = (tmp1 * th <= tmp2) .* (tmp2 * th <= tmp1);
    min_map = mask .* min_map + (~mask);
end

if (scale_con)
    th = 0.4;
    tmp1 = repmat(((region_data1(:,2).*region_data1(:,1))./(region_data1(:,5).*region_data1(:,4))), [1 size(region_data2, 1)]);
    tmp2 = repmat(((region_data2(:,2).*region_data2(:,1))./(region_data2(:,5).*region_data2(:,4)))', [size(region_data1, 1) 1]);
    mask = (tmp1 * th <= tmp2) .* (tmp2 <= tmp1/th);
    min_map = mask .* min_map + (~mask);    
end