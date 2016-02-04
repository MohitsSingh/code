function [ imout ] = gradientMagnitude(im)

% GRADIENTNORM 
% using sobel operator to calculate the gradient norm

sx = [-1 0 1; -2 0 2; -1 0 1];
sy = [1 2 1; 0 0 0; -1 -2 -1];

tic
imout = conv2(im, sx, 'same') + conv2(im, sy, 'same');
toc