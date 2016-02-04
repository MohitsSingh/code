%% This checks the invariance of learned network to difference image formations,
%% as well as the cross-style performance.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;


exp_suffix = '_none';
cnn_cifar_mod('modelType','lenet','exp_suffix',exp_suffix);

exp_suffix = '_grayscale';
dataFun = @(x) repmat(rgb2gray(uint8(x)),1,1,3);
cnn_cifar_mod('modelType','lenet','exp_suffix',exp_suffix,'dataFun',dataFun);



