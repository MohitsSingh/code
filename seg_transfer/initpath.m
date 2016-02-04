baseDir = '~/code/';
addpath(genpath(fullfile(baseDir,'/3rdparty/SelectiveSearchPcode')));
addpath('/home/bagon/develop/berkeley_seg/grouping/lib');
addpath('~/code/3rdparty');
addpath('~/code/3rdparty/uri/');

addpath(genpath('/home/amirro/code/3rdparty/libsvm-3.12'));

% vlfeat
addpath('~/code/3rdparty/vlfeat-0.9.14/toolbox');
vl_setup;

% PASCAL VOC
addpath('/home/amirro/data/VOCdevkit/VOCcode/');
VOCinit

% Color descriptors...
addpath(genpath('/home/amirro/code/3rdparty/colordescriptors30/'));
colorDescPath = '/home/amirro/code/3rdparty/colordescriptors30/x86_64-linux-gcc/colorDescriptor';


% textons...
% addpath(genpath('/home/amirro/code/3rdparty/proposals'));

col = load(fullfile('/home/amirro/code/3rdparty/proposals/classifiers', 'colorClusters.mat'));
tex = load(fullfile('/home/amirro/code/3rdparty/proposals/classifiers', 'textonClusters.mat'));


% globalOpts.expPath = fullfile(experimentPath,'%s.mat');
% choose a subset of training & testing images.
%     [train_images,test_images] = getdatasets(VOCopts,globalOpts);

fid = fopen(sprintf(VOCopts.imgsetpath,VOCopts.trainset));
train_images = textscan(fid,'%s');
train_images = train_images{1};
fclose(fid);

fid = fopen(sprintf(VOCopts.imgsetpath,VOCopts.testset));
test_images = textscan(fid,'%s');
test_images = test_images{1};
fclose(fid);

