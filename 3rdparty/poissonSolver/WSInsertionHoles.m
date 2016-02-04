im = double(imread('../images/wall001.tif', 'TIF'));
figure;
imshow(mat2gray(im));

iminsert = double(imread('../images/words.tif', 'TIF'));
figure;
imshow(mat2gray(iminsert));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

boxSrc = [5 195 5 195 ];
posDest = [15 15];
imr = poissonSolverInsertionHoles(imir, imr, boxSrc, posDest);
img = poissonSolverInsertionHoles(imig, img, boxSrc, posDest);
imb = poissonSolverInsertionHoles(imib, imb, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), '../images/wallResult.jpg', 'JPG');
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));