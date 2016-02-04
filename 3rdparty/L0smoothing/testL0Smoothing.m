Im = imread('/home/amirro/code/3rdparty/exemplarsvm/images/VOC2007/009704.jpg');
S = L0Smoothing(Im,0.01);
figure, imshow(S);
figure,imshow(Im);
