% function fcnTrain_action_objects_8_classes_full_pascal(varargin)
%FNCTRAIN Train FCN model using MatConvNet
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_classes_full_pascal_zoom' ;
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

%load ~/storage/misc/pascal_action_imdb.mat
load ~/storage/misc/pascal_imdb_only_my_actions.mat
for t = 1:1:length(imdb.labels)
    L = imdb.labels{t};
%     if max(L(:)) >1
%         clf; subplot(1,2,1); imagesc2(L);colorbar
        L = LUT(L+1)-1;
%         subplot(1,2,2); imagesc2(L);colorbar
%         dpc
%     end
    imdb.labels{t} = L;
end

% imdb = pascal_imdb;
imdb.images = 1:length(imdb.images_data);
% train = find(imdb.isTrain);
% val = find(~imdb.isTrain);

images_data = {};
labels = {};
isTrain = {};
for t = 1:length(imdb.images_data)
    img = imdb.images_data{t};
    L = imdb.labels{t};
    curBoxes = imdb.img_boxes{t};
    z = true(size2(img));
    for iBox = 1:size(curBoxes,1)
        bb = curBoxes(iBox,:);
        bb = makeSquare(bb);
        bb = round(inflatebbox(bb,1.2,'both',false));
        curSubImg = uint8(cropper(img,bb));
        curLabels = cropper(L,bb);
        curLabels = cropper(L,bb);
        %         x2(curLabels)
        toCare = cropper(z,bb);
        if (mod(t,50)==0)
            clf; subplot(1,2,1); imagesc2(curSubImg);
            subplot(1,2,2); imagesc2(curLabels);
            dpc(.001)
        end
        %         dpc;
        curLabels(~toCare) = 255;
        images_data{end+1} = curSubImg;
        labels{end+1} = curLabels;
    end
    isTrain{end+1}= ones(size(curBoxes,1),1)*imdb.isTrain(t);
end

isTrain = cat(1,isTrain{:});
imdb.isTrain = isTrain;
imdb.images_data = images_data;
imdb.labels = labels;
train = find(imdb.isTrain);
val = find(~imdb.isTrain);


%     labels = {};
% classLabels = {'none','face','hand',...
%     'phoning',...
%     'playinginstrument',...
%     'reading',...
%     'ridingbike',...
%     'ridinghorse',...
%     'running',...
%     'takingphoto',...
%     'usingcomputer',...
%     'walking'};
imdb.nClasses = 7; % 
% save(opts.imdbPath,'imdb','train','val','test');


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

