imgPath = '/home/amirro/storage/data/ILSVRC2012/images/train/n04523525/n04523525_10114.JPEG';


% gm convert

I = single(imread(imgPath));
sz = size(I);
factor = max(256./sz(1:2));
opts.interpolation = 'bilinear';
I_piotr = imResample(I, factor, 'bilinear');
I_matlab = imresize(I, ...
    'scale', factor, ...
    'method', opts.interpolation) ;


figure(1);
imshow(I/255);title('original image');
figure(2); 
imshow(I_piotr/255);title('piotr bilinear');
figure(3); 
imshow(I_matlab/255); title('matlab bilinear');
I_pre = imread(strrep(imgPath,'ILSVRC2012','ILSVRC2012_pre')); 
figure(4); imshow(I_pre);
title('GraphicsMagick (Lanczos)');
I_matlab_lanc_2 = imresize(I, ...
    'scale', factor, ...
    'method', 'lanczos2') ;
figure(5); 
imshow(I_matlab_lanc_2/255); title('matlab lanczos2');
I_matlab_lanc_3 = imresize(I, ...
    'scale', factor, ...
    'method', 'lanczos3') ;
figure(6); 
imshow(I_matlab_lanc_3/255); title('matlab lanczos3');