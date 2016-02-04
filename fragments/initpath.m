baseDir = '~/code/';
addpath(genpath(fullfile(baseDir,'/3rdparty/SelectiveSearchPcode')));
addpath('/home/bagon/develop/berkeley_seg/grouping/lib');
addpath('~/code/3rdparty');
addpath('~/code/3rdparty/uri/');




addpath(genpath('/home/amirro/code/3rdparty/libsvm-3.12'));

% local
addpath(genpath('features'));
addpath('utils');
addpath('parallel');
addpath('boxes')
addpath('vis');
addpath('training');
addpath('config');

% vlfeat 
addpath('~/code/3rdparty/vlfeat-0.9.14/toolbox');
vl_setup;

% PASCAL VOC 
addpath('/home/amirro/data/VOCdevkit/VOCcode/');

% Color descriptors...
addpath(genpath('/home/amirro/code/3rdparty/colordescriptors30/'));
colorDescPath = '/home/amirro/code/3rdparty/colordescriptors30/x86_64-linux-gcc/colorDescriptor';

% textons...
% addpath(genpath('/home/amirro/code/3rdparty/proposals'));
