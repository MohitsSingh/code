
% startup;
load('models/face_1_final.mat');
imname = '/home/amirro/data/Stanford40/JPEGImages/drinking_194.jpg';
% load and display image
im = imread(imname);
% im = imresize(im,2);
% im = imrotate(im,-40,'bilinear','crop');
clf;
image(im);
axis equal; 
axis on;
% disp('input image');
% disp('press any key to continue'); pause;
% disp('continuing...');

% [boxes, boxes_r, bboxes, info] = my_imgdetect_r(im, model,-2);

[ds, bs] = imgdetect(im, model,-2);
top = nms(ds, 0.5);
clf;
% if model.type == model_types.Grammar
%   bs = [ds(:,1:4) bs];
% end
showboxes(im, ds(1,:));
