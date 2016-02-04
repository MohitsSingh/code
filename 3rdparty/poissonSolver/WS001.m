% im = double(imread('../images/orange.tif', 'TIF'));
% figure;
% imshow(mat2gray(im));
% 
% iminsert = double(imread('../images/orange.tif', 'TIF'));
% figure;
% imshow(mat2gray(iminsert));


im = double(imread('../images/monalisaBW.tif', 'TIF'));

% find the area of the white pixels
n = size(find(im))




% im = double(imread('../images/test001.tif', 'TIF'));
% figure;
% imshow(mat2gray(im));
% 
% iminsert = double(imread('../images/test001.tif', 'TIF'));
% figure;
% imshow(mat2gray(iminsert));
% 
% 
% [imr img imb] = decomposeRGB(im);
% [imir imig imib] = decomposeRGB(iminsert);
% 
% boxSrc = [20 190 20 190 ];
% posDest = [20 20];
% imr = poissonSolver(imir, imr, boxSrc, posDest);
% img = poissonSolver(imig, img, boxSrc, posDest);
% imb = poissonSolver(imib, imb, boxSrc, posDest);
% 
% imnew = composeRGB(imr, img, imb);
% 
% figure(100);
% imshow(mat2gray(imnew));
% imwrite(mat2gray(imnew), '../images/test001Result.jpg', 'JPG');
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));