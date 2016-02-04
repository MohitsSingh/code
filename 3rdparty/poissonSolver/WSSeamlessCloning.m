im = double(imread('../images/orange.tif', 'TIF'));
figure;
imshow(mat2gray(im));

iminsert = double(imread('../images/orange.tif', 'TIF'));
figure;
imshow(mat2gray(iminsert));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

boxSrc = [100 190 200 290 ];
posDest = [100 100];
imr = poissonSolver(imir, imr, boxSrc, posDest);
img = poissonSolver(imig, img, boxSrc, posDest);
imb = poissonSolver(imib, imb, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), '../images/orangeResult.jpg', 'JPG');
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));