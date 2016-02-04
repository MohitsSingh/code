% external directories

% vl-feat
vlfeatPath = '~/code/3rdparty/vlfeat-0.9.16/toolbox';
addpath(vlfeatPath);
vl_setup;

addpath('~/code/3rdparty/GCMex');

% parameters
VOCPath = '/home/amirro/storage/VOCdevkit/VOCcode';
addpath(VOCPath);
VOCinit;

addpath('util');
addpath('perf');
addpath('graph');
dataDir = 'data';
ensuredir('data')
ensuredir('data/gists');
ensuredir('data/bow');
ensuredir('data/superpix');

addpath('gist');
addpath('bow');