function cnn_imagenet_ratio_10(varargin)
% CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%   This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%   VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'))
opts.dataDir = fullfile('/home/amirro/storage/data/','ILSVRC2012') ;
opts.modelType = 'vgg-f' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile('/home/amirro/storage/data/', sprintf('imagenet12-%s-%s-ratio_10', ...
                                       sfx, opts.networkType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = 2 ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir ;
if ~opts.batchNormalization
  opts.train.learningRate = logspace(-2, -4, 60) ;
else
  opts.train.learningRate = logspace(-1, -4, 20) ;
end
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                   Database initialization
% -------------------------------------------------------------------------

if exist(opts.imdbPath)  
  imdb = load(opts.imdbPath) ;
else
  %imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  orig_imdb_path = '/home/amirro/storage/data/imagenet12-vgg-f-bnorm-simplenn/imdb.mat';
  imdb = load(orig_imdb_path);    
  sets = imdb.images.set;  
  sel_ = false(size(sets));
  sel_(sets ~= 1) = true;
  f = find(sets==1);
  sel_(f(1:10:end)) = true;
  imdb.images.id = imdb.images.id(sel_);
  imdb.images.name = imdb.images.name(sel_);
  imdb.images.set = imdb.images.set(sel_);
  imdb.images.label = imdb.images.label(sel_);
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end
imdb.got_image = false(size(imdb.images.set));
imdb.images_cache = cell(1,length(imdb.images.set));

% movefile /home/amirro/storage/data/imagenet_2012_imdb.mat /home/amirro/storage/data/imagenet12-alexnet-bnorm-simplenn/imdb.mat
% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

net = cnn_imagenet_init('model', opts.modelType, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod', opts.weightInitMethod) ;
bopts = net.normalization ;
bopts.numThreads = opts.numFetchThreads ;

% compute image statistics (mean, RGB covariances etc)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% One can use the average RGB value, or use a different average for
% each pixel
%net.normalization.averageImage = averageImage ;
net.normalization.averageImage = rgbMean ;

switch lower(opts.networkType)
  case 'simplenn'
  case 'dagnn'
    net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
    net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1error') ;
  otherwise
    error('Unknown netowrk type ''%s''.', opts.networkType) ;
end

% -------------------------------------------------------------------------
%                                               Stochastic gradient descent
% -------------------------------------------------------------------------

[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v' ;
useGpu = numel(opts.train.gpus) > 0 ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = getBatchSimpleNNWrapper(bopts) ;
    %[net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', true) ;
    % AMIR CHANGED CONSERVE MEMORY FROM TRUE TO FALSE
    [net,info] = cnn_train(net, imdb, fn, opts.train, 'conserveMemory', false) ;
  case 'dagnn'
    fn = getBatchDagNNWrapper(bopts, useGpu) ;
    opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
    info = cnn_train_dag(net, imdb, fn, opts.train) ;
end

% -------------------------------------------------------------------------
function fn = getBatchSimpleNNWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchSimpleNN(imdb,batch,opts) ;

% -------------------------------------------------------------------------
function [im,labels] = getBatchSimpleNN(imdb, batch, opts)
% -------------------------------------------------------------------------
image_paths = strcat([imdb.imageDir filesep], imdb.images.name(batch));
images_to_load = ~imdb.got_image(batch);
image_paths = image_paths(images_to_load);

if nargout==0 % prefetch
    vl_imreadjpeg(image_paths, 'numThreads', opts.numThreads, 'Prefetch','Preallocate',false);
else
    loadedImages = vl_imreadjpeg(image_paths,'numThreads', opts.numThreads);
    for t = 1:length(loadedImages)
        if size(loadedImages{t},3)>3
            warning('found more than 3 channels in input image...');
            loadedImages{t} = loadedImages{t}(:,:,1:3);
        end
        loadedImages{t} = uint8(loadedImages{t});
    end
    imdb.got_image(batch) = true;
    imdb.images_cache(batch(images_to_load)) = loadedImages;
%     profile on
    im = cnn_imagenet_get_batch_and_remember(imdb.images_cache(batch), opts, ...
                                'prefetch', nargout == 0) ;
%     profile viewer
    labels = imdb.images.label(batch) ;
end

% -------------------------------------------------------------------------
function fn = getBatchDagNNWrapper(opts, useGpu)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatchDagNN(imdb,batch,opts,useGpu) ;

% -------------------------------------------------------------------------
function inputs = getBatchDagNN(imdb, batch, opts, useGpu)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
im = cnn_imagenet_get_batch(images, opts, ...
                            'prefetch', nargout == 0) ;
if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, opts)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
fn = getBatchSimpleNNWrapper(opts) ;
for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{t} = mean(temp, 4) ;
  rgbm1{t} = sum(z,2)/n ;
  rgbm2{t} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
