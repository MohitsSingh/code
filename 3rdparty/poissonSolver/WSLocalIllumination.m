im = double(imread('../images/orange001.tif', 'TIF'));
figure;
imshow(mat2gray(im));

iminsert = double(imread('../images/orange001.tif', 'TIF'));
figure;
imshow(mat2gray(iminsert));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

boxSrc = [100 190 100 190 ];
posDest = [100 100];
imr = poissonSolverLocalIllumination(imir, imr, boxSrc, posDest);
img = poissonSolverLocalIllumination(imig, img, boxSrc, posDest);
imb = poissonSolverLocalIllumination(imib, imb, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), '../images/orangeResult2.jpg', 'JPG');
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));