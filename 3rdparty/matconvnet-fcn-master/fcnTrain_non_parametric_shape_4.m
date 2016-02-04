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

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-non_parametric_shape/theta_scale2' ;
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
opts.train.numSubBatches = 1 ;
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
%%
baseImage = zeros(7);
baseImage(4,:) = 1;
baseImage = imResample(baseImage,[224 224],'nearest');

nImages = 500;
%rotations = pi*(rand(nImages,1)-.5)/6;
rotations = rand(nImages,1)*180-90;
r = 10^(log(8));
% rotations = 0*rotations;

scaleBins = 2.^linspace(-1,1,8);  %  rand(nImages,2)*30-15;
% scaleBins = 1;
scales = rand(size(rotations))*(max(scaleBins)-min(scaleBins))+min(scaleBins);

% bin the output space...
thetaBins = linspace(-90,90,30);
[~,thetaLabels] = histc(rotations ,thetaBins);
[~,scaleLabels] = histc(scales ,scaleBins);

nRots = length(thetaBins);
nScales = length(scaleBins);
nClasses = nRots*nScales;

% generate some random images from these parameters
train = 1:400;
val = 401:500;
imgSize = filt_size;
images = {};
ps = {};
%%

labels = zeros(1,1,2,nImages,'single');
for t = 1:nImages
    t
%     iRot = thetaLabels(t);
%     jScale = scaleLabels(t);
    I = xform(baseImage,rotations(t),scales(t),224);        
    labels(:,:,:,t) =  [rotations(t)/180,log(scales(t))];%[ sub2ind([nRots nScales],iRot,jScale);
    images{t} = im2uint8(cat(3,I,I,I));
%     clf; imagesc2(I);axsis on
%     dpc,continue   
end

%%

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
%%
imdb.images_data = images;
imdb.images.set = [ones(1,400),2*ones(1,100)];
imdb.labels = labels;%reshape(labels,1,1,[],1);
imdb.nClasses = nClasses;
gpuDevice(1);
% Get initial model from VGG-VD-16
% opts.nClasses = imdb.nClasses;
% net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;

opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-f.mat'
net = fcnInitializeModel_parametric_prediction_theta2('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses);
  
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
%%
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
opts.train.batchSize = 20;
%%
opts.train.learningRate = 1e-5; % was 1e-4
% Launch SGD
% opts.train = rmfield(opts.train,'trainingOrder');
getBatchWrapper = @(x) @(imdb,batch) getBatch_p_theta_simple(imdb,batch,x,'prefetch',nargout==0) ;
opts.train.errorFunction = 'pdist';
opts.train.save_ratio = 5;
info = cnn_train(net, imdb, getBatchWrapper(bopts), opts.train, ...
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
modelPath = fullfile(opts.expDir,'net-epoch-90.mat');
net = load(modelPath) ;
net = net.net;
net.layers = net.layers(1:end-1);
net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;

net.mode = 'test' ;


predVar = 'x20';
inputVar = 'input' ;
imageNeedsToBeMultiple = true 
net.move('gpu');
addpath('~/code/3rdparty/subplot_tight');
%% (for the basis : layer 33)
%%
% imagesc(squeeze(net.params(34).value(:,:,:,2)))
%%

%%
for imId = 1:1:500
    imId
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
%     rgb = rgb+abs(.1*randn(size(rgb)));
    
    curLabels = single(imdb.labels(imId));
    net.meta.normalization.averageImage = reshape(stats.rgbMean,1,1,3);
    
    net.eval({inputVar, gpuArray(rgb)}) ;
    scores = gather(net.vars(net.getVarIndex(predVar)).value);
    figure(1);
    clf; subplot(1,2,1); imagesc2(rgb); title('orig')
    subplot(1,2,2);
    VV = zeros(size(rgb));
    VV(:,:,1) = rgb(:,:,1);
    curRot = scores(1)*180;
    curScale = exp(scores(2));
    rgb_xformed = xform(baseImage,curRot,curScale,224);
    VV(:,:,2) = single(rgb_xformed(:,:,1));
    imagesc2(VV);
    
  
    pause
end

%%

