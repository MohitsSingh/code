function [ imout ] = stitchSeam( im )

% stitch the 4 sides of the given image patch togather
% top to bottom, left to right

[height width] = size(im);

top = im(1, :);
bottom = im(height, :);

left = im(:, 1);
right = im(:, width);

imout = zeros(height, width);

imout(1, :) = (top + bottom)/2;
imout(height, :) = imout(1, :);
imout(:, 1) = (left + right)/2;
imout(:, width) = imout(:, 1);

% figure;
% imshow(mat2gray(imout));