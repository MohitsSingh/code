addpath('/home/amirro/code/3rdparty/vlfeat-0.9.18/toolbox');
addpath('/home/amirro/code/3rdparty/sc');
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
addpath(genpath('/home/amirro/code/utils/'));
addpath('/home/amirro/code/common');
addpath('/home/amirro/code/3rdparty/uri/');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/matlab');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/matlab');


vl_setup;
vl_setupnn
conf.net = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/imagenet-vgg-f.mat');
voc_devkit_path = '/home/amirro/storage/data/voc07/VOCdevkit/VOCdevkit/VOCcode';
addpath(voc_devkit_path);
VOCinit



