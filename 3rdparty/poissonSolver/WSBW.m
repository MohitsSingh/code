im = double(imread('../images/test001BW.tif', 'TIF'));
figure;
imshow(mat2gray(im));

iminsert = double(imread('../images/test001BW.tif', 'TIF'));
figure;
imshow(mat2gray(iminsert));


% [imr img imb] = decomposeRGB(im);
% [imir imig imib] = decomposeRGB(iminsert);


boxSrc = [20 100 20 100];
posDest = [20 20];
imnew = poissonSolver(iminsert, im, boxSrc, posDest);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), '../images/test001BWResult.jpg', 'JPG');

% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));