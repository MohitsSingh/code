% function fcnTrain_action_objects_classes_subpatches(varargin)
%FNCTRAIN Train FCN model using MatConvNet

addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath utils;

% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-fcnTrain_hands' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;
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
opts.train.numEpochs=100;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
% load the imdb for training full images, then sample windows around the
% predicted regions to refine them. 
%%
if exist(opts.imdbPath)
  load(opts.imdbPath) ;
else
    
  baseDataDir = '/home/amirro/storage/data/hand_dataset/';
  training_dir = fullfile(baseDataDir,'training_dataset/training_data');
  addpath('/net/mraid11/export/data/amirro/data/hand_dataset/training_dataset');
  curDir = pwd;
  cd /net/mraid11/export/data/amirro/data/hand_dataset/training_dataset
  training_data = getTrainingData();
  cd(curDir);
  
    
  imdb.images_data = {training_data.img};
  imdb.labels = {training_data.anno};
  train = 1:length(imdb.labels);
  val = 1:5:length(train);
  train  = setdiff(train,val);
  
  
%   train = train(1:10:end);
%   val = val(1:10:end);        
  imdb.class_labels = {'none','hand'}; 
  
  imdb.nClasses = 1;
  
  newIMDBPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-fcnTrain_action_objects_classes_subpatches/imdb.mat';
  save(opts.imdbPath,'imdb','train','val');
end
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

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
opts.train.gpus = 1;
getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj(imdb,batch,x,'prefetch',nargout==0) ;
% Launch SGD
opts.train.prefetch = false;
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

%%
%% test the performance....

test_params.labels =   {'none','hand'};
test_params.labels_to_block = [];
test_params.prefix = 'perfs_ap';
test_params.set = 'val';
test = [];
[perfs,diags] = test_net_perf(opts.expDir,17,imdb,train,val,test,test_params);

figure,plot(diags(:,2))