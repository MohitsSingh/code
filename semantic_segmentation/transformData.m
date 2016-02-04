function [ data_t ] = transformData( imgs,mean_pix,img_size )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

for t = 1:length(imgs)
    if (isa(imgs{t},'double') || isa(imgs{t},'single')) && max(imgs{t}(:))<=1
        imgs{t} = 255*imgs{t};
    end
end

MM = multiResize(imgs,img_size);
batchdata = single(cat(4,MM{:}));
% im = all_images{1};
% batchdata = single(all_images{1});
for c = 1:3
    batchdata(:, :, c, :) = batchdata(:, :, c, :) - mean_pix(c);
end
if size(batchdata,3)>3
    batchdata=batchdata(:,:,[3 2 1 4:size(batchdata,3)],:);
else
    batchdata=batchdata(:,:,[3 2 1],:);
end
data_t=permute(batchdata,[2 1 3 4]);

end

