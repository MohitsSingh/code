% Main function for action recognition demo.
startup;
addpath '~/code/mircs';

s40PathImagesPath = '';
experiment.basePath = s40PathImagesPath;
experiment.dbtype = 'stanford40';
experiment.datadir = '~/storage/s40';
experiment.class_sel = {'drinking'};
imdb = createIMDB(experiment);

all_data = preprocess(conf,imdb); % preprocess all images (extract faces, landmarks, etc)

% phase 1: training
train_classifiers(conf,all_data)



cd /home/amirro/code/3rdparty/edgeBoxes/
%I = imread('/home/amirro/code/3rdparty/DeepPyramid-master/000084.jpg');
%%
I = imread('~/data/Stanford40/JPEGImages/applauding_112.jpg');
clf;
x2(I)
% I = imcrop(I);
%%
I2 = imResample(I,1,'bilinear');
opts.minBoxArea = 100;
tic, bbs=edgeBoxes(I2,model,opts); toc
% 
bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
x2(I2);r = getSingleRect;r(3:4) = r(1:2)+r(3:4);
% x2(I); plotBoxes(r);
ovps = boxesOverlap(bbs(:,1:4),r(1:4));
[k,ik] = sort(ovps,'descend');
figure(1)
size(bbs)
for iu = 1:size(bbs,1)
    u = ik(iu);
    clf;imagesc2(I2);
    plotBoxes(bbs(u,:));
    drawnow;
    pause
end

% x2(I)
%