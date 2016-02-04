% function fcnTrain_non_parametric_shape
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

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-non_parametric_shape' ;
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
opts.train.numEpochs=101;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% generate a basis for the input images.
baseImages = {};
filt_size = 224;
T = zeros(filt_size,'single');
T(:,end/3:2*end/3) = 1;
thetas = 0:20:160;
nBase = length(thetas);
for iR = 1:nBase
    r = thetas(iR);
    baseImages{end+1} = imrotate(T,r,'bicubic','crop');
end
x2(baseImages);

baseImages = cat(4,baseImages{:});

% generate some random images from these shapes:
% for each image, choose a subset of the filters,
% then filter with a sparse matrix.
nImages = 150;
train = 1:100;
val = 101:50;
imgSize = 224;
images = {};

for t = 1:nImages
    p = randperm(nBase);
    p = p(1);
    z = rand(length(p),1);
    z = z/sum(z);
    I = zeros(imgSize,'single');
    for u = 1:length(z)
        I = I+baseImages(:,:,:,p(u));
    end
    images{t} = im2uint8(cat(3,I,I,I));
end

% for t = 1:nImages
%     t
%     p = randperm(nBase);
%     p = p(1:3);
%     M = imgSize+filt_size-1;
%     I = zeros(imgSize,imgSize,1,1);
%     
%     B = [];
%     locs = randi(M,3,2);
%     for u = 1:3
%         X = zeros(M,M,1,1,'single');
%         X(locs(u,1),locs(u,2)) = 1;
%         X = vl_nnconv(gpuArray(X),gpuArray(baseImages(:,:,p(u))),[]);
%         I = I+gather(X);
%     end
%    
% end

labels = cellfun2(@(x) uint8(x(:,:,3) > .5),images);
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

imdb.nClasses=1;
imdb.images_data = cellfun2(@(x) im2uint8(min(1,max(0,x))),images);
imdb.images.set = [ones(1,100),2*ones(1,50)];
imdb.labels = labels;

% Get initial model from VGG-VD-16
% opts.nClasses = imdb.nClasses;
net = fcnInitializeModel_parametric_prediction('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
opts.modelType = 'fcn16s';
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
net.meta.classes = {'obj'};
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

getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj(imdb,batch,x,'prefetch',nargout==0) ;

info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;
%% look at some results.

modelPath = fullfile(opts.expDir,'net-epoch-1.mat');
net = load(modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
net.mode = 'test' ;
for name = {'objective', 'accuracy'}
    net.removeLayer(name) ;
end

predVar = 'precition';
inputVar = 'input' ;
imageNeedsToBeMultiple = true 
net.move('gpu');
addpath('~/code/3rdparty/subplot_tight');
%%
imId=1;
rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
%rgb = imResample(rgb,2*[384 384],'bilinear');

curLabels = single(imdb.labels{imId});
net.meta.normalization.averageImage = reshape(stats.rgbMean,1,1,3);
[pred,scores] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,'prediction',rgb+rand(size(rgb))*0,1,true,{'none','obj'});
softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));

