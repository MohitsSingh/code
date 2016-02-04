% function fcnTrain_probs_to_action(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8_probs_to_action' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;

% opts.sourceModelPath = '/net/mraid11/export/data/amirro//fcn/data/models/imagenet-vgg-verydeep-16.mat' ;
% opts.sourceModelPath = '/net/mraid11/export/data/amirro/matconv_data/pascal-fcn8s-tvg-dag.mat';

% opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
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
opts.train.gpus = 1;%[1 2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.numEpochs=100;
% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
    load(opts.imdbPath) ;
else
    load ~/code/mircs/fra_db_2015_10_08.mat
    outPath = '~/storage/fra_action_fcn';
    clear LL;
    LL.scores_coarse = {};
    LL.scores_fine = {};
    for t = 1:1:length(fra_db);
        %     if fra_db(t).isTrain,continue,end
        t
        p = j2m(outPath,fra_db(t));
        L = load(p);
        coarse_probs = L.scores_full_image;
        fine_probs = L.scores_hires;
        coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
        LL.scores_coarse{t} = coarse_probs;
        LL.scores_fine{t} = fine_probs;
    end
    
    imdb.images = 1:length(fra_db);
    imdb.images_data = LL.scores_coarse;
    
    mean_img = mean(cellfun3(@(x) mean(reshape(x,[],8),1),imdb.images_data(train),1),1);
    
    isTrain = [fra_db.isTrain];
    train = find(isTrain);
    val = train(1:3:end);
    test = find(~isTrain);
    train = setdiff(train,val);
    imdb.images = 1:length(isTrain);
    
    imdb.labels = [fra_db.classID];
    imdb.nClasses = 8;
    %%
    %%%%% initialize network
    
    %imdb_old = imdb;               
    %%%%%
    %%save ~/storage/misc/LL.mat LL -v7.3    
    imdb.images_orig = imdb.images_data;    
%     tt = [train(:);val(:)];
%     images_data = {};
    for tt = 1:length(imdb.images_data)        
        imdb.images_data{tt} = LL.scores_coarse{tt};
    end    
    
%     
%     z = zeros(100);
%     z(50,50)=1;
%     z = imdilate(z,ones(5));
%     imagesc2(z)
    old_labels = imdb.labels;
    for t =898:length(imdb.images)
        t
        m = old_labels{t};
        m(m<3) = 0;
        m = m-2;
        m = imdilate(m,ones(11));
        imdb.labels{t} = m;
%         clf; imagesc2(imdb.labels{t});
%          dpc
    end        
    imdb.nClasses = 5;
    
    
    for u = 1:length(imdb.labels)
        m = imdb.labels{u};
        if any(m(:)==2)
            clf; imagesc2(imResample(m,[100 100],'nearest'))
            dpc
        end
    end
    
    displayImageSeries2(imdb.labels(1:10:end))    
%     imdb = imdb_old;
    %%
    %
    gpuDevice(1)
    %%
    clear net;
    net = cnn_for_probs(); 
    %%
    % -------------------------------------------------------------------------
    % Train
    % -------------------------------------------------------------------------        
%     profile on    
    opts.train.learningRate = 0.0001 * ones(1,50) ;
    % Setup data fetching options
    bopts.numThreads = opts.numFetchThreads ;
    bopts.labelStride = 1 ;
    opts.train.batchSize = 5;
    bopts.labelOffset = 1 ;
    bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
    bopts.rgbMean = mean_img;
    bopts.useGpu = numel(opts.train.gpus) > 0 ;
    %getBatchWrapper = @(x) @(imdb,batch) getBatch_class_to_action(imdb,batch,x,'prefetch',nargout==0) ;
    getBatchWrapper = @(x) @(imdb,batch) getBatch_class_to_action(imdb,batch,x,'prefetch',nargout==0) ;
    % Launch SGD
    info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
        'train', train, ...
        'val', val) ;
    profile viewer
    % -------------------------------------------------------------------------
    