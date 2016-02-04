function [ I1 ] = normalize_image( I,net )
%NORMALIZE_IMAGE Summary of this function goes here
%   Detailed explanation goes here
% obtain and preprocess an image
% I = rgb2gray(I);
% I = cat(3,I,I,I);
I = single(I) ; % note: 255 range

I1 = zeros([net.normalization.imageSize size(I,4)],'single');
for t = 1:size(I,4)
    I1(:,:,:,t) = imResample(I(:,:,:,t), net.normalization.imageSize(1:2)) ;
end
I1 = bsxfun(@minus,I1,net.normalization.averageImage);

