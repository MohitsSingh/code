%imgPath = 'X:\data\101087.jpg';
imgPath = 'Person-of-Interest-Root-Cause-Episode-13-5.jpg';
I = imread(imgPath);

sigma_ = .5;
c = 5;
minSize = 1000;
R =  mexFelzenSegmentIndex(I, sigma_, 'RGI', minSize);

CC = seg2col(I,R);

figure,imshow(CC,[]);
figure,imshow(I)

figure,imshow(edge(rgb2gray(im2single(I)),'canny'))


% B = colfilt(I,[5 5],'distinct',@boundary_func);
% 
% figure,imshow(B)
% figure,imshow(I)
% 
% figure,imshow(edge(I,'canny'));
% [R,F] = vl_mser(I);
% 
% F_ = vl_ertr(F);
% figure,imshow(I);
% hold on;
% vl_plotframe(F_);
% 
