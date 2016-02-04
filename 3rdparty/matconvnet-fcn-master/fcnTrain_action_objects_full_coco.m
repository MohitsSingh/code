% function fcnTrain_action_objects_full_coco(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath('utils/');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcnTrain_action_objects_full_coco' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s';
% opts.sourceModelPath = '/net/mraid11/export/data/amirro//fcn/data/models/imagenet-vgg-verydeep-16.mat' ;
% opts.sourceModelPath = '/net/mraid11/export/data/amirro/matconv_data/pascal-fcn8s-tvg-dag.mat';
opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
% [opts, varargin] = vl_argparse(opts, varargin) ;
% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = [];%fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '12' ;
opts.vocAdditionalSegmentations = true ;
opts.numFetchThreads = 1 ; % not used yet
% training options (SGD)
opts.train.batchSize = 20 ;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = 2;%[1 2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.numEpochs=101;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% if exist(opts.imdbPath)
%   load(opts.imdbPath) ;
% else
% initialize coco
addpath(genpath('~/code/utils'));
addpath(genpath('/home/amirro/code/3rdparty/coco-master/MatlabAPI'));
%dataDir='../';
dataDir='~/storage/mscoco'
dataType='train2014';
annFile=sprintf('%s/annotations/instances_%s.json',dataDir,'train2014');
if(~exist('coco_train','var')), coco_train=CocoApi(annFile); end

annFile=sprintf('%s/annotations/instances_%s.json',dataDir,'val2014');
if(~exist('coco_val','var')), coco_val=CocoApi(annFile); end

train_anno_ids = [coco_train.data.annotations.id];
val_anno_ids = [coco_val.data.annotations.id];

train = coco_train.inds.imgIds;
val = coco_val.inds.imgIds;
imdb.images_ids = [train;val];
imdb.images = 1:length(imdb.images_ids);
imdb.coco_train = coco_train;
imdb.set = single([ones(size(train));2*ones(size(val))]);
imdb.coco_val = coco_val;


%imdb.nClasses = length(coco_train.data.categories);
imdb.nClasses = max([coco_train.data.categories.id]);

imdb.dataDir = '~/storage/mscoco';

%   save(opts.imdbPath,'imdb','train','val','test');
% end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
%%
% Get initial model from VGG-VD-16
opts.nClasses = imdb.nClasses;
net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
  % upgrade model to FCN16s
  net = fcnInitializeModel16s(net) ;
end
if strcmp(opts.modelType, 'fcn8s')
  % upgrade model fto FCN8s
  net = fcnInitializeModel8s(net) ;
end
stats.rgbMean =[116.6725  111.5917  103.1466]';
net.meta.normalization.rgbMean = stats.rgbMean;
% 
%%
% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj_coco(imdb,batch,x,'prefetch',nargout==0) ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', 1:length(train), ...
  'val', 1:length(val)) ;


