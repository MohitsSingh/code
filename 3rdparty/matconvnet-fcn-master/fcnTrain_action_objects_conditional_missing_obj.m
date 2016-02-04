
function net = my_fcn_train(imdb,net_name,nEpochs,other_opts)

function fcnTrain_action_objects_conditional_missing_obj(varargin)
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

baseDir = '/net/mraid11/export/data/amirro/fcn/data/';
opts.expDir = fullfile(baseDir,net_name);% fcn8s-action_obj_conditional_missing_obj' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;
opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
[opts, varargin] = vl_argparse(opts, varargin) ;
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
opts.train.numEpochs=nEpochs;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------% Get initial model from VGG-VD-16
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
net.meta.classes = {'face','object'};

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

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch_action_obj(imdb,batch,opts,'prefetch',nargout==0) ;

    
% 
% function fcnTrain_action_objects_conditional_missing_obj(varargin)
% %FNCTRAIN Train FCN model using MatConvNet
% 
% run matconvnet/matlab/vl_setupnn ;
% addpath(genpath('~/code/utils'));
% addpath(genpath('~/code/3rdparty/piotr_toolbox'));
% addpath('~/code/3rdparty/sc');
% addpath('utils/');
% addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
% vl_setup
% 
% run matconvnet/matlab/vl_setupnn ;
% addpath matconvnet/examples ;
% % experiment and data paths
% 
% opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_conditional_missing_obj' ;
% opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
% mkdir(opts.expDir);
% mkdir(opts.dataDir);
% opts.modelType = 'fcn8s' ;
% % opts.sourceModelPath = '/net/mraid11/export/data/amirro//fcn/data/models/imagenet-vgg-verydeep-16.mat' ;
% % opts.sourceModelPath = '/net/mraid11/export/data/amirro/matconv_data/pascal-fcn8s-tvg-dag.mat';
% 
% opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
% [opts, varargin] = vl_argparse(opts, varargin) ;
% % experiment setup
% opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;
% opts.imdbStatsPath = [];%fullfile(opts.expDir, 'imdbStats.mat') ;
% opts.vocEdition = '12' ;
% opts.vocAdditionalSegmentations = true ;
% opts.numFetchThreads = 1 ; % not used yet
% % training options (SGD)
% opts.train.batchSize = 20 ;
% opts.train.numSubBatches = 10 ;
% opts.train.continue = true ;
% opts.train.gpus = 2;%[1 2];
% opts.train.prefetch = true ;
% opts.train.expDir = opts.expDir ;
% opts.train.learningRate = 0.0001 * ones(1,50) ;
% opts.train.numEpochs = numel(opts.train.learningRate) ;
% opts.train.numEpochs=100;
% opts = vl_argparse(opts, varargin) ;
% 
% % -------------------------------------------------------------------------
% % Setup data
% % -------------------------------------------------------------------------
% 
% if exist(opts.imdbPath)
%   load(opts.imdbPath) ;
% else
%   load ~/code/mircs/images_and_face_obj_full.mat;
%   
%   sel_ = [fra_db.classID] == 2;
%   fra_db = fra_db(sel_);
%   isTrain = isTrain(sel_);
%   images = images(sel_);
%   masks = masks(sel_);
%   train = find(isTrain);
%   val = train(1:3:end);
%   test = find(~isTrain);
%   train = setdiff(train,val);
%   imdb.images = 1:length(images);
%   imdb.images_data = images;
%   needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
%   if needToConvert
%       for t = 1:length(images)
%           imdb.images_data{t} = im2uint8(imdb.images_data{t});
%       end
%   end  
%   for t = 1:length(masks)
%       m = masks{t};      
%       z = m == 0;
%       obj = m == 1;
%       m = m-1;
%       m(z) = 0;
%       m(obj) = 0;
% %       m(m>=3) = 3; % TODO - THIS IS DONE TO MAKE THE ACTION OBJECT A SINGLE CLASS
%       masks{t} = m;
% %       clf; subplot(1,2,1); imagesc2(masks{t}); colorbar;
% %       subplot(1,2,2); imagesc2(m); colorbar;
% %       dpc
%   end    
%   masks = cellfun2(@uint8,masks);
%   imdb.labels = masks;
%   imdb.nClasses = 3;
%   save(opts.imdbPath,'imdb','train','val','test');
% end
% 
% % -------------------------------------------------------------------------
% % Setup model
% % -------------------------------------------------------------------------
% 
% % Get initial model from VGG-VD-16
% opts.nClasses = imdb.nClasses;
% net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
% if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
%   % upgrade model to FCN16s
%   net = fcnInitializeModel16s(net) ;
% end
% if strcmp(opts.modelType, 'fcn8s')
%   % upgrade model fto FCN8s
%   net = fcnInitializeModel8s(net) ;
% end
% stats.rgbMean =[116.6725  111.5917  103.1466]';
% net.meta.normalization.rgbMean = stats.rgbMean;
% % 
% net.meta.classes = {'face','object'};
% 
% % -------------------------------------------------------------------------
% % Train
% % -------------------------------------------------------------------------
% % Setup data fetching options
% bopts.numThreads = opts.numFetchThreads ;
% bopts.labelStride = 1 ;
% bopts.labelOffset = 1 ;
% bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
% bopts.rgbMean = stats.rgbMean ;
% bopts.useGpu = numel(opts.train.gpus) > 0 ;
% 
% % Launch SGD
% info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
%   'train', train, ...
%   'val', val) ;
% 
% 
% %% test it a bit...
% perfs = struct('cm',{},'cm_n',{},'n_epoch',{})
% ppp = 1:1:100;
% for ipp = 1:length(ppp)
%     pp = ppp(ipp);
%     modelPath = ['/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_conditional_missing_obj/net-epoch-' num2str(pp) '.mat'];    
%     [perfs(ipp).cm,perfs(ipp).cm_n] = test_segmentation_accuracy(modelPath,imdb,train,val,test,{'none','face','hand','obj'});
%     perfs(ipp).n_epoch = pp;
% %     modelPath = ['/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_conditional_no_help/net-epoch-' num2str(pp) '.mat'];
% %     test_segmentation_accuracy(modelPath,imdb,train,val,test,{'none','obj'});
% end
% i
% diags = {};
% for t = 1:length(perfs)
%     diags{t} = diag(perfs(t).cm_n);
% end
% 
% diags = cat(2,diags{:})';
% %plot(diags(1:5:end,:));legend('bg','face','hand','obj');
% plot(diags(:,:));legend('bg','face','hand','obj');
% 
% %% try to learn only subsets: only hands, only objects, only faces, etc.
% 
% opts.train.numEpochs = 40;
% imdb_orig = imdb;
% class_sel = [];
% objs = {'face','hand','obj'};
% obj_sel = {};
% 
% for iSel = 1:length(obj_sel)
%     curObjs = obj_sel{iSel};
%     curName = '...';
%     curNet = train_net_for_subset('....');
%     measureNetPerformance();
% end
% 
% 
% 
% 
% % -------------------------------------------------------------------------
% function fn = getBatchWrapper(opts)
% % -------------------------------------------------------------------------
% fn = @(imdb,batch) getBatch_action_obj(imdb,batch,opts,'prefetch',nargout==0) ;
