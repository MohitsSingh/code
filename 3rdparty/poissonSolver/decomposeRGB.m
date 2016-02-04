function [ imr, img, imb ] = decomposeRGB( im )
% decompose a color image (RGB) into its seperate RGB components


imr = im(:, :, 1);
img = im(:, :, 2);
imb = im(:, :, 3);