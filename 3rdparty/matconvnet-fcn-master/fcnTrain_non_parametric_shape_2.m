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

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-non_parametric_shape/relu' ;
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


% generate some random images from these shapes:
% for each image, choose a subset of the filters,
% then filter with a sparse matrix.
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
imgSize = filt_size;
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

labels = cellfun2(@(x) uint8(x(:,:,3) > .5),images);


% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
%%
imdb.nClasses=1;
imdb.images_data = images;
imdb.images.set = [ones(1,100),2*ones(1,50)];
imdb.labels = labels;
gpuDevice(2);
% Get initial model from VGG-VD-16
% opts.nClasses = imdb.nClasses;
% net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
net = fcnInitializeModel_parametric_prediction('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses);
  
opts.modelType = 'fcn32s';
% opts.modelType = 'fcn8s';
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
opts.train.batchSize = 5;
% Launch SGD


getBatchWrapper = @(x) @(imdb,batch) getBatch_parametric(imdb,batch,x,'prefetch',nargout==0) ;
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
  'train', train, ...
  'val', val) ;
%% look at some results.


%%

G = {};
T = 0;

L = containers.Map('KeyType', 'char', 'ValueType', 'char');
for iLayer = 1:length(net.layers)
    L(net.layers(iLayer).outputs{1}) =  net.layers(iLayer).name;
end

L('input') = 'input';
L('label') = 'label';
for iLayer = 1:length(net.layers)
    curLayer = net.layers(iLayer);
    inputs = curLayer.inputs;
    outputs = curLayer.outputs;
    
    for ii = 1:length(inputs)
        for jj = 1:length(outputs)
            T = T+1;
            G{T,1} = L(inputs{ii});
            G{T,2} = L(outputs{jj});
%             G{T,1} = (inputs{ii});
%             G{T,2} = (outputs{jj});
        end
    end
end

graphName = 'mygraph';
dotFilePath = '~/fcn.dot'
writeDotFile(graphName,G,[],dotFilePath);
graphImagePath = sprintf('%s.jpg',dotFilePath(1:end-4));
cmd = sprintf('dot -Tjpg %s -o %s',dotFilePath,graphImagePath);
system(cmd)
imshow(graphImagePath)

%%

%/net/mraid11/export/data/amirro/fcn/data/fcn8s-non_parametric_shape/

%modelPath = fullfile(opts.expDir,'f_32_basis/net-epoch-101.mat');
modelPath = fullfile(opts.expDir,'net-epoch-33.mat');
net = load(modelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
net.mode = 'test' ;
for name = {'objective', 'accuracy'}
    net.removeLayer(name) ;
end

predVar = 'prediction';
inputVar = 'input' ;
imageNeedsToBeMultiple = true 
net.move('gpu');
addpath('~/code/3rdparty/subplot_tight');
%% (for the basis : layer 33)
imagesc(net.params(33).value(:,:,:,5))
%%
% imagesc(squeeze(net.params(34).value(:,:,:,2)))
%%
% x2(net.params(33).value(:,:,2))
%%
% % extract alexnet features.
% net = load('~/storage/matconv_data/imagenet-vgg-f.mat');
% im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
% im_ = im_ - net.normalization.averageImage ;
% net = vl_simplenn_move(net,'gpu');
% % run the CNN
% im_ = repmat(im_,1,1,1,256);
% tic
% res = vl_simplenn(net, gpuArray(im_)) ;
% toc
% 
% net = dagnn.DagNN.loadobj(load(modelPath)) ;
% im = imread('peppers.png') ;
% im_ = single(im) ; % note: 255 range
% im_ = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
% im_ = im_ - net.meta.normalization.averageImage ;
% net.eval({'data', im_}) ;
% 
% % show the classification result
% scores = squeeze(gather(net.vars(end).value)) ;


%%
for imId = 1:5:150
rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});

% rgb = imerode(rgb,ones(50));

%rgb = imResample(rgb,2*[384 384],'bilinear');


curLabels = single(imdb.labels{imId});
net.meta.normalization.averageImage = reshape(stats.rgbMean,1,1,3);
[pred,scores] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,'prediction',rgb+rand(size(rgb))*0,1,true,{'none','obj'});
softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
figure(3), imagesc(softmaxScores(:,:,2))
figure(4), imagesc(pred);
figure(5), imagesc(softmaxScores(:,:,2)>.73)

dpc
end

%%

