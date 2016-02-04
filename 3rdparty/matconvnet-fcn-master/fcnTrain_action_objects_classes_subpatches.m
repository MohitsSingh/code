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
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-fcnTrain_action_objects_classes_subpatches' ;
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
  load /net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/imdb.mat    
  
  full_net_model_path= '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
  [net_full,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(full_net_model_path, 'matconvnet');
  net_full.move('gpu');

  class_labels = {'none','face','hand','drink','smoke','blow','brush','phone'};
  imdb_full = imdb;
%   train_set = union(train,val);
%   train_set = setdiff(1:1215,train_set);

  imgs = {};
  labels = {};    
  train_val = [train val];
  train_val_inds = {};
  for it = 1:1:length(train_val)
      it
      k = train_val(it);
      curImage = imdb_full.images_data{k};
      imgOrig = curImage;
      curLabels = imdb_full.labels{k};
      bw_dontcare =  bwdist(curLabels > 0)>40;
      curLabels(bw_dontcare) = 255;
      sz_orig = size2(curImage);
      curImage = imResample(single(curImage),[384 384],'bilinear');            
      [pred,scores_] = predict_and_show(net_full,imageNeedsToBeMultiple,inputVar,'prediction',curImage,1,false,class_labels);                 
      pred = imResample(pred,sz_orig,'nearest');
      scores_ = imResample(scores_,sz_orig,'nearest');
      softmaxScores = bsxfun(@rdivide,exp(scores_),sum(exp(scores_),3));      
      M = max(softmaxScores(:,:,2:end),[],3);      
      [subs,vals] = nonMaxSupr( double(M), 20, .2,5 );
      subs = fliplr(subs);
      boxes = round(inflatebbox(subs,mean(sz_orig)/3,'both',true));
      patchesDontCare = multiCrop2(bw_dontcare,boxes);
      dontCareAreaRatio = cellfun3(@(x) nnz(x)/numel(x),patchesDontCare);
      boxes(dontCareAreaRatio>.5,:) = [];
      if (isempty(boxes))
          fprintf('warning: no good patches found for image %d\n,skipping',k); 
          continue
      end
%       bw_dontcare = cat(3,bw_dontcare,bw_dontcare,bw_dontcare);
%       imgOrig(bw_dontcare) = 255;
      curPatches = multiCrop2(imgOrig,boxes);                              
      imgs{it} = curPatches;
      labels{it} = multiCrop2(curLabels,boxes);
      T = ones(1,length(curPatches));      
      if it >length(train)
          T = 2*T;
      end
      train_val_inds{it} = T;
%       figure(1);clf; imagesc2(sc(cat(3,M,im2single(imgOrig)),'prob')); plotPolygons(subs,'g+');
%       plotBoxes(boxes);
%       figure(2); mImage(curPatches);
%       dpc; continue
  end
  
  imgs = [imgs{:}];
  labels = [labels{:}];
  
%   for t = 1:50:length(imgs)
%       clf; subplot(1,2,1); imagesc2(imgs{t});
%       subplot(1,2,2); imagesc2(double(labels{t}).*(labels{t}~=255));
%       dpc
%   end
%   
  train_val_inds = [train_val_inds{:}];
  
  train = find(train_val_inds==1);
  val = find(train_val_inds==2);
  
  imdb.images = 1:length(imgs);
  imdb.images_data = imgs;
  imdb.labels = labels;
%   x2(cellfun2(@(x) repmat(single(x)/8,1,1,3),labels))
  
  % generate only training and validation, test will be there later.      
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
getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj(imdb,batch,x,'prefetch',nargout==0) ;
% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;
