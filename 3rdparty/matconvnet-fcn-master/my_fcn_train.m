
function net = my_fcn_train(imdb,net_name,nEpochs,other_opts,startNet)

% function fcnTrain_action_objects_conditional_missing_obj(varargin)
%FNCTRAIN Train FCN model using MatConvNet


train = imdb.train;
val = imdb.val;
test = imdb.test;
% experiment and data paths
if nargin < 4
    other_opts = struct('gpus',1);%,'freeze',[]);
end
if ~isfield(other_opts,'gpus')
    other_opts.gpus=1;
end
% if ~isfield(other_opts,'freeze')
%     other_opts.freeze=[];
% end
if ~isfield(other_opts,'extra_data')
    other_opts.extra_data = [];
end

if ~isfield(other_opts,'lr_ratio')
    other_opts.lr_ratio =1;
end


if ~isfield(other_opts,'resetGPU')
    other_opts.resetGPU = true;
end
if ~isfield(other_opts,'modelType')
    other_opts.modelType = 'fcn8s' ;
end
if nargin < 5
    startNet = [];
end

baseDir = '/net/mraid11/export/data/amirro/fcn/data/';
opts.expDir = fullfile(baseDir,net_name);% fcn8s-action_obj_conditional_missing_obj' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = other_opts.modelType;
opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
% experiment setup
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
opts.imdbStatsPath = [];%fullfile(opts.expDir, 'imdbStats.mat') ;
opts.vocEdition = '12' ;
opts.vocAdditionalSegmentations = true ;
opts.numFetchThreads = 1 ; % not used yet
% training options (SGD)
opts.train.batchSize = 20 ;
opts.train.resetGPU = other_opts.resetGPU;
opts.train.numSubBatches = 10 ;
opts.train.continue = true ;
opts.train.gpus = other_opts.gpus;
% opts.train.freeze = other_opts.freeze;
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;
opts.train.learningRate = opts.train.learningRate/other_opts.lr_ratio;
if isfield(other_opts,'lr_ratio')
    opts.train.learningRate = opts.train.learningRate*other_opts.lr_ratio;
end
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.numEpochs=nEpochs;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------% Get initial model from VGG-VD-16
stats.rgbMean =[116.6725  111.5917  103.1466]';
if isempty(startNet)
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
else
    net = startNet;
end
if ~isfield(net.meta.normalization,'rgbMean')
    net.meta.normalization.rgbMean = stats.rgbMean;
end

%
% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
bopts.rgbMean = net.meta.normalization.rgbMean; %stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', train, ...
    'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch_action_obj(imdb,batch,opts,'prefetch',nargout==0) ;
