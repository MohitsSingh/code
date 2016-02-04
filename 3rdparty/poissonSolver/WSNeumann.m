dir = '../images/neumann/';
filename = 'test001';
ext = '.tif';

im = double(imread(strcat(dir, filename, ext)));
figure;
imshow(mat2gray(im));

iminsert = double(imread(strcat(dir, filename, ext)));
figure;
imshow(mat2gray(iminsert));

[imr img imb] = decomposeRGB(im);
[imir imig imib] = decomposeRGB(iminsert);

boxSrc = [100 190 100 190 ];
posDest = [100 100];
imr = poissonSolverNeumann(imir, imr, boxSrc, posDest);
img = poissonSolverNeumann(imig, img, boxSrc, posDest);
imb = poissonSolverNeumann(imib, imb, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(100);
imshow(mat2gray(imnew));
imwrite(mat2gray(imnew), strcat(dir, filename, 'R' , ext));
% poisson1(50, 51, 5);


% im = double(imread('../images/test001BW.tif', 'TIFF'));
% figure;
% imshow(mat2gray(im));