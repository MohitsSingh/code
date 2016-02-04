function [ result ] = gradientNorm(im, box)

% GRADIENTNORM 
% calculate the gradient norm inside the box

if nargin == 1
    [height width] = size(im);
    x0 = 1;
    x1 = width;
    y0 = 1;
    y1 = height;
else
	x0 = box(1);
	x1 = box(2);
	y0 = box(3);
	y1 = box(4);
	
	width = x1-x0+1;
	height = y1-y0+1;
end;
    
imout = gradientMagnitude(im);
% imout
% 
% figure;
% imshow(mat2gray(imout));

result =  sum(sum(abs(imout(x0:x1, y0:y1))))/(width*height);