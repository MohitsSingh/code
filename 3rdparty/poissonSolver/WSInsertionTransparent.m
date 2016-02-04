im = double(imread('../images/tropicalIsland.tif', 'TIF'));
figure;
imshow(mat2gray(im));

iminsert = double(imread('../images/rainbow001.tif', 'TIF'));
figure;
imshow(mat2gray(iminsert));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

boxSrc = [2 160 2 256 ];
posDest = [2 2];
imr = poissonSolverInsertionHoles(imir, imr, boxSrc, posDest);
img = poissonSolverInsertionHoles(imig, img, boxSrc, posDest);
imb = poissonSolverInsertionHoles(imib, imb, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), '../images/rainbowResult.jpg', 'JPG');
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));