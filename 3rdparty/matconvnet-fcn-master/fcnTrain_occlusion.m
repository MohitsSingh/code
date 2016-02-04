function fcnTrain_occlusion(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
% experiment and data paths


opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-occlusion' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;
%opts.sourceModelPath = '/net/mraid11/export/data/amirro//fcn/data/models/imagenet-vgg-verydeep-16.mat' ;
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
opts.train.gpus = 1;%[1 2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% if exist(opts.imdbPath)
  %imdb = load(opts.imdbPath) ;
  
 % load ~/code/mircs/images_and_face_obj.mat;
 
  % create polygons...
  
  
  
  
  imdbPath = 'polys.mat';
  if exist(imdbPath,'file')
      load(imdbPath);
  else  
    addpath('~/code/3rdparty');
    imgSize = [200 200];
    
    nLayers = 2;
    nImages = 100;
    nSides = 10;
%     isTrain = false(nImages,1);
%     isTrain(1:nImages*.8) = true;
%     7
    train = 1:round(nImages*.6);
    val = train(end)+1:round(nImages*.7);
    test = val(end)+1:nImages;
    
    imdb.images = 1:nImages;
    imdb.images_data = {};
    imdb.labels = {};
    imdb.nClasses = nLayers;
    for t = 1:nImages
        label = zeros(imgSize);
        img = zeros(imgSize);
        for iLayer = 1:nLayers
            [x,y,dt] = simple_polygon(nSides);
            x = x+.5;
            y = y+.5;
            x = x*imgSize(2)/2;
            y = y*imgSize(1)/2;            
            r = poly2mask2([x y],[200 200]);
            label(r) = iLayer;
            img(label==iLayer) = rand();
        end
        
        imdb.images_data{t} = im2uint8(repmat(img,[1 1 3]));
        imdb.labels{t} = label;
%         figure(1);
%         clf;
%         subplot(1,2,1);imagesc2(img); title('img');
%         subplot(1,2,2);imagesc2(label);title('label');
%         dpc
    end
    save(imdbPath,'imdb','train','val','test');
  end
  
  

%   train= find(isTrain);
%   val = train(1:3:end);
%   train = setdiff(train,val);  
  
%   imdb.images_data = images;
  
%   imdb.labels = masks;
%   imdb.nClasses = 2;

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
