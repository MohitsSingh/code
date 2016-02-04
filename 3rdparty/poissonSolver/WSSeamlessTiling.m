
name = 'pebble';
ext = '.jpg';
dir = '../images/tile/';


infile = strcat(dir, name, ext);


im = double(imread(infile));
figure;
imshow(mat2gray(im));

[imr img imb] = decomposeRGB(im);

%----------------------------------------------
% construct the 4 edges
%----------------------------------------------
[height width] = size(imr);

imr_new = stitchSeam(imr);
img_new = stitchSeam(img);
imb_new = stitchSeam(imb);

% pre-allocate tiling plane
hFill = 2;
wFill = 3;
canvas = zeros(hFill*height, wFill*width, 3);
canvas_new = canvas;


boxSrc = [2 width-1 2 height-1];
posDest = [2 2];

imr = poissonSolver(imr, imr_new, boxSrc, posDest);
img = poissonSolver(img, img_new, boxSrc, posDest);
imb = poissonSolver(imb, imb_new, boxSrc, posDest);

imnew = composeRGB(imr, img, imb);

figure(50);
imshow(mat2gray(imnew));
outfile = strcat(dir, name, '_new.jpg');
imwrite(mat2gray(imnew), outfile);

%-----------------------------------------------
% seamless tiling
%-----------------------------------------------

for x = 1:wFill
    for y = 1:hFill
        y0 = (y-1)*height+1;
        y1 = y0+height-1;
        x0 = (x-1)*width +1;
        x1 = x0+width-1;
        
        canvas_new(y0:y1, x0:x1, :) = imnew;
    end
end

figure(100);
imshow(mat2gray(canvas_new));
outfile = strcat(dir, name, '_newT.jpg');
imwrite(mat2gray(canvas_new), outfile);

%-----------------------------------------------
% seamful tiling
%-----------------------------------------------

for x = 1:wFill
    for y = 1:hFill
        y0 = (y-1)*height+1;
        y1 = y0+height-1;
        x0 = (x-1)*width +1;
        x1 = x0+width-1;
        
        canvas(y0:y1, x0:x1, :) = im;
    end
end

figure(200);
imshow(mat2gray(canvas));
outfile = strcat(dir, name, '_oldT.jpg');
imwrite(mat2gray(canvas), outfile);
