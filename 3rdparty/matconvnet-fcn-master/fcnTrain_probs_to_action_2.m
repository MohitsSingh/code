% function fcnTrain_probs_to_action(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('/home/amirro/storage/pascal_parts/');
addpath(genpath('~/code/utils'));
addpath('utils');
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8_probs_to_action_2' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;
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


imdb_gt_labels = load('/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/imdb.mat');
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
    
       
    isTrain = [fra_db.isTrain];
    train = find(isTrain);
    val = train(1:3:end);
    test = find(~isTrain);
    train = setdiff(train,val);
    imdb.images = 1:length(isTrain);
    
    mean_img = mean(cellfun3(@(x) mean(reshape(x,[],8),1),imdb.images_data(train),1),1);       
    imdb.labels = [fra_db.classID];
    imdb.nClasses = 8;
end

imdb.labels = imdb_gt_labels.imdb.labels;

%%
for t = 1:50:1000
    curImg = imdb_gt_labels.imdb.images_data{t};
    pred = imdb.labels{t};All files (*)
    scores_ = pred;
    showPredictions_dummy(single(curImg),pred,scores_,class_labels,1);
    dpc
end
%
% sz = size(fine_probs);
% net.eval({'input', fine_probs});
% scores = gather(net.vars(net.getVarIndex('prediction')).value);
%% initialize network...
%%

gpuDevice(1)


% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
%     profile on
%clear net;
%%
labels_bkp = imdb.labels;
%%
% we don't care about anything but the action object!! 
for t = 1:length(imdb.labels)
    t
    curLabel = labels_bkp{t};
%     curLabel(curLabel <= 2) = 0;
%     curLabel = imdilate(curLabel,ones(3));
    imdb.labels{t} = curLabel;
end

%%
net = cnn_for_probs();

opts.train.learningRate = 0.001 * ones(1,50) ; % was 1/10 x 
% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
opts.train.batchSize = 10; % was 5
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
bopts.classWeights(1) = .1;
bopts.rgbMean = mean_img;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
imdb.nClasses = 7;
%getBatchWrapper = @(x) @(imdb,batch) getBatch_class_to_action(imdb,batch,x,'prefetch',nargout==0) ;
getBatchWrapper = @(x) @(imdb,batch) getBatch_class_to_action(imdb,batch,x,'prefetch',nargout==0) ;
% Launch SGD


info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', train, ...
    'val', val) ;
% profile off
% -------------------------------------------------------------------------

%%


%%
curModelPath = fullfile(opts.expDir,'net-epoch-45.mat');
net = load(curModelPath) ;
net = dagnn.DagNN.loadobj(net.net) ;
net.meta.normalization.averageImage = reshape(mean_img,1,1,[]) ;
predVar = net.getVarIndex('prediction') ;
inputVar = 'input' ;
net.move('gpu');
% 
% 

%%
addpath('~/code/3rdparty/subplot_tight');
%%
class_labels ={'none','face','hand','drink','smoke','blow','brush','phone'};
for iVal = 30:5:length(val)
    curData = imdb.images_data{val(iVal)};
    [scores,pred] = applyNet(net,curData,false,'input','prediction');
    %clf; figure(1); imagesc2(mImage(curData));
    scores = imResample(scores,size2(curData));
    pred = imResample(pred,size2(curData));
    softmaxScores = bsxfun(@rdivide,exp(scores),sum(exp(scores),3));
    
    showPredictions(single(imdb_gt_labels.imdb.images_data{val(iVal)}),pred,softmaxScores,class_labels,1);
    dpc
end
% 
% [net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);
