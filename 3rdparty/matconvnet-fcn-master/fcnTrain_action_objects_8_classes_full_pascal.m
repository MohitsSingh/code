% function fcnTrain_action_objects_8_classes_full_pascal(varargin)
%FNCTRAIN Train FCN model using MatConvNet
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
addpath('/home/amirro/code/3rdparty/');
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_classes_full_pascal' ;
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
opts.train.gpus = 1;%[1 2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.00001 * ones(1,50) ; % was 0.0001
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.numEpochs=101;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
cats_we_like = [0 0 1 0 1 0 0 0 1 1 0]';


if (0)
    load ~/storage/misc/pascal_action_imdb.mat
    imdb = pascal_imdb;
    imdb.images = 1:length(imdb.images_data);
    train = find(imdb.isTrain);
    val = find(~imdb.isTrain);
    
    % now remove all images with no labels at all and find the unique set of
    % labels.
    load ~/storage/misc/action_names
    my_cats = {'phoning','reading','takingphoto','usingcomputer'};
    [lia,lib] = ismember(my_cats,action_names)
    %z_cat = false(size(action_names));
    %z_cat(lib) = true;
    z_cat = cats_we_like;
    imdb_old = imdb;
    sel_ = false(size(imdb.labels));
    for t = 1:1:length(imdb.labels)
        if any(max(imdb.img_label_vecs{t},[],2) & z_cat)
            sel_(t) = true;
            %        clf; subplot(1,2,1);imagesc2(imdb.images_data{t});
            %        subplot(1,2,2); imagesc2(imdb.labels{t})
            %        dpc(.1)
        end
    end
    
    
    imdb.images_data  = imdb_old.images_data(sel_);
    imdb.labels  = imdb_old.labels(sel_);
    imdb.img_boxes  = imdb_old.img_boxes(sel_);
    imdb.isTrain  = imdb_old.isTrain(sel_);
    imdb.imageIds  = imdb_old.imageIds(sel_);
    imdb.img_labels  = imdb_old.img_labels(sel_);
    imdb.img_label_vecs  = imdb_old.img_label_vecs(sel_);
    imdb.img_box_ids = imdb_old.img_box_ids(sel_);
    imdb.images = imdb_old.images(sel_);
    train = find(imdb.isTrain);
    val = find(~imdb.isTrain);
    imdb.images = 1:length(imdb.images);
    
    % find the unique set of labels
    u = {};
    for t = 1:length(imdb.labels)
        t
        u{t} = unique(imdb.labels{t}(:));
    end
    u = cat(1,u{:});
    u = unique(u);
    
    % unique labels 
    
    
    
    
    
    
%     LUT = 0:length(u)-1;
    
    LUT = zeros(1,max(u));
    LUT(u+1) = 1:length(u)
    
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
    
    
    save ~/storage/misc/pascal_imdb_only_my_actions.mat imdb train val LUT my_cats cats_we_like -v7.3
end
load ~/storage/misc/pascal_imdb_only_my_actions.mat
  1
    2
    3
    4
    5
    6
   10
   11
   z = 0;
% make a dummy thingy  
for t = 1:1:length(imdb.labels)
    L = imdb.labels{t};    
    L = LUT(L+1)-1;
%     if any(L(:)>=3)
        t
%         imdb.ima
%         z = z+1;
%         clf; subplot(1,2,1); imagesc2(imdb.images_data{t});colorbar
       
%         subplot(1,2,2); imagesc2(L);colorbar
%         dpc
%     end
    imdb.labels{t} = L;
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

% % Launch SGD
getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj(imdb,batch,x,'prefetch',nargout==0) ;

info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', train, ...
    'val', val) ;

