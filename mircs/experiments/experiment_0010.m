%%%%%% Experiment 9 %%%%%%%
% Nov. 7, 2013

% run the ramanan code to get keypoints on AFLW. (do this in the SGE dir)

initpath;
config;
addpath('~/code/3rdparty/face-release1.0-basic/'); % zhu & ramanan
L_imgs = load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');

a = detect_landmarks(conf,L_imgs.ims(101),2);

mImage(L_imgs.ims(1:100));