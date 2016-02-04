addpath('/home/amirro/code/3rdparty/vlfeat-0.9.14/toolbox');
addpath(genpath('/home/amirro/code/3rdparty/libsvm-3.12'));
addpath('/home/amirro/code/3rdparty/ssim');
addpath('/home/amirro/data/VOCdevkit/VOCcode/');

DPM_path = '/home/amirro/code/3rdparty/voc-release4.01/';

vl_setup;

% vl_setup;
% addpath(genpath('D:\libsvm-3.12'));

opts.hands_locs_suff = 'hands_locs';
opts.hands_images_suff = 'hands_imgs';

% uncomment the following line if you wish to run the labeling tool.
%labeling_script;

% change this directory to where you put your standford40 dataset
%inputDir = 'D:\Stanford40\';
inputDir = '/home/amirro/data/Stanford40/JPEGImages';
imageSplitDir = '/home/amirro/data/Stanford40/ImageSplits';

ext = '.jpg';

actionsFileName = '/home/amirro/data/Stanford40/ImageSplits/actions.txt';
[A,ii] = textread(actionsFileName,'%s %s');

f = fopen(actionsFileName);
A = A(2:end);