function [net, info] = cnn_imagenet_with_orig_data_w_cmyk_color_fix(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
%  This demo demonstrates training the AlexNet, VGG-F, VGG-S, VGG-M,
%  VGG-VD-16, and VGG-VD-19 architectures on ImageNet data.

addpath(genpath('~/code/3rdparty/piotr_toolbox'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = fullfile(vl_rootnn, 'data','ILSVRC2012') ;
% opts.dataDir = '/home/amirro/storage/data/ILSVRC2012_pre/';
opts.modelType = 'alexnet' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile('/home/amirro/storage/data/', ['imagenet12-' sfx]);
opts.expDir = [opts.expDir '_orig_data_w_cmyk_color_fix'];
%opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = cnn_imagenet_init('model', opts.modelType, ...
                        'batchNormalization', opts.batchNormalization, ...
                        'weightInitMethod', opts.weightInitMethod, ...
                        'networkType', opts.networkType) ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_imagenet_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

imdb.imageDir = '/home/amirro/storage/data/ILSVRC2012/images';
% remove the bad images...
load bad_imgs.mat;
% 
is_cmyk = ismember(imdb.images.name,bad_imgs);
imdb.images.is_cmyk = is_cmyk;

% toKeep = ~ismember(imdb.images.name,bad_imgs);
% imdb.images.id = imdb.images.id(toKeep);
% imdb.images.set = imdb.images.set(toKeep);
% imdb.images.name = imdb.images.name(toKeep);
% imdb.images.label = imdb.images.label(toKeep);
% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.description ;

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the image average (use either an image or a color)
%net.meta.normalization.averageImage = averageImage ;
net.meta.normalization.averageImage = rgbMean ;

% Set data augmentation statistics
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
clear v d ;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train_cmyk_first ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end
% opts.train.gpus = 2;
% 
% 
% fid = fopen('~/bad_imagenet_images','r');
% line = 0;
% bad_files = {};
% while(line~=-1)
%     line = fgetl(fid);
%      if (line == -1)
%         break
%      end
%          
%     %line = strrep(line,'/home/amirro/storage/data/ILSVRC2012_pre/images/','');
%     bad_files{end+1} = line;
%    
% end
% fclose(fid)


% profile off
opts.train.gpus = 2;
gpuDevice(2);
opts.train.profile = 0;
opts.train.prefetch = true;
opts.train.conserveMemory=false;
% profile on
net.meta.trainOpts.numEpochs = 40;
[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      net.meta.trainOpts, ...
                      opts.train) ;
% profile viewer
% profile viewer
% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')

switch opts.networkType
  case 'simplenn'
    save(modelPath, '-struct', 'net') ;
  case 'dagnn'
    net_ = net.saveobj() ;
    save(modelPath, '-struct', 'net_') ;
    clear net_ ;
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case 'dagnn'
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch_color_fix(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch_color_fix(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.label(batch) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'input', im, 'label', imdb.images.label(batch)} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;

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
