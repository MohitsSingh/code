dir = '../images/ns/';
filename = 'monalisa1';
ext = '.tif';


im = double(imread(strcat(dir, filename, ext)));
figure;
imshow(mat2gray(im));

iminsert = double(imread(strcat(dir, filename, 'I3', ext)));
figure;
imshow(mat2gray(iminsert));

imMask = double(imread(strcat(dir, filename, 'M', ext)));
figure;
imshow(mat2gray(imMask));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

offset = [0 0];
imr = poissonSolverSeamlessCloning1(imir, imr, imMask, offset);
img = poissonSolverSeamlessCloning1(imig, img, imMask, offset);
imb = poissonSolverSeamlessCloning1(imib, imb, imMask, offset);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), strcat(dir, filename, 'R', ext));
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));