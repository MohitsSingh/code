
im = imread('../images/test_small.tif', 'TIF');

tic
dx = [1 -2 1];
dy = dx';

imt = conv2(dx, im);
imt = conv2(dy, imt);
figure;
imshow(mat2gray(imt));

toc

imt

tic
l = [0 1 0; 1 -4 1; 0 1 0];
imt1 = conv2(l, im);

figure;
imshow(mat2gray(imt1));

toc

imt1