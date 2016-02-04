%%% Experiment 0042  %%%%%%%%%%%%%%%%%
%%% 15/6/2014
%% Train DPM for different human poses to predict location of hands. 

initpath;config;

h3dPath = '/home/amirro/storage/poselets/data/person/image_sets/h3d/';
load(fullfile(h3dPath,'h3d.mat'));
% vocDir = '/home/amirro/storage/data/VOC2010/VOCdevkit/VOC2010';
% poseDir = '/home/amirro/storage/data/VOC2010/info';

imgDir = fullfile(h3dPath,'images');
figure,plot()
(a_h3d_train.image_id)