% function [net, info] = my_cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
    vl_setup
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');
addpath(genpath('~/code/utils'));

opts.modelType = 'lenet' ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

switch opts.modelType
    case 'lenet'
        opts.train.learningRate = [0.05*ones(1,15) 0.005*ones(1,10) 0.0005*ones(1,5)] ;
        opts.train.weightDecay = 0.0001 ;
    case 'nin'
        opts.train.learningRate = [0.5*ones(1,30) 0.1*ones(1,10) 0.02*ones(1,10)] ;
        opts.train.weightDecay = 0.0005 ;
    otherwise
        error('Unknown model type %s', opts.modelType) ;
end
opts.expDir = fullfile('data', sprintf('cifar-%s', opts.modelType)) ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.numEpochs = numel(opts.train.learningRate) ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','cifar') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.train.batchSize = 100 ;
opts.train.continue = true ;
opts.train.gpus = 1 ;
opts.train.expDir = opts.expDir ;
% opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                               Prepare data and model
% --------------------------------------------------------------------

imdb = load(opts.imdbPath) ;
%% Now, check several options for training order.
opts.expDir = 'data/cifar-lenet';
% First option : random training order.
net = cnn_cifar_init(opts) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% do some benchmarking
[recalls,precisions,F_scores]  = doEvaluation_simple( opts,imdb,1,30,'normal');

%% Second option (silly option) : group training by classes:

%% Now, check several options for training order.
opts.expDir = 'data/cifar-lenet_class_by_class';
opts.train.expDir = opts.expDir;
ensuredir(opts.expDir);
% First option : random training order.
net = cnn_cifar_init(opts) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.trainingOrder = 'class_by_class';
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% do some benchmarking
[recalls,precisions,F_scores]  = doEvaluation_simple( opts,imdb,1,30,'normal');


%% sort by each of examples : do the easiest examples first
imdb_for_testing = imdb;
old_set = imdb_for_testing.images.set;
LUT = [3 0 1];
imdb_for_testing.images.set = LUT(old_set);
% do some benchmarking
opts.expDir = 'data/cifar-lenet';

[recalls,precisions,F_scores]  = doEvaluation_simple( opts,imdb_for_testing,30,30,'normal_eval_on_train');

R = load('data/cifar-lenet/perfnormal_eval_on_train/epoch_30.mat');

% when is an example easy? rank "easyness" according to mean precision on
% example.


perms_per_class = zeros(size(R.curBatchResults));

for t = 1:10
    
    class_scores = R.curBatchResults(t,:);
    class_labels = 2*(R.curBatchLabels==t)-1;
    [scores, perm] = sort(class_scores, 'descend') ;
    
    [recall,precision] = vl_pr(class_labels,class_scores);
    
    perms_per_class(t,:) = perm;
    
    % sanity: get the top k highest precision samples for this class and
    % compare their precision-recall to the overall precision-recall curve.
%     
%     figure(1); vl_pr(class_labels,class_scores);    
%     %[v,iv] = sort(precs_per_class(t,:),'descend');
%     iv = perm;
%     K = 1000;
%     figure(2); vl_pr(class_labels(iv(1:K)), class_scores(iv(1:K)));
%     
end


imdb = load(opts.imdbPath);
f_train = find(imdb.images.set==1);
f_test = find(imdb.images.set==3);

mean_perm = mean(perms_per_class);
[v,iv] = sort(mean_perm,'ascend');
f_train = f_train(iv);
new_order = [f_train f_test];
imdb.images.data = imdb.images.data(:,:,:,new_order);
imdb.images.labels = imdb.images.labels(new_order);
imdb.images.set = imdb.images.set(new_order);


opts.expDir = 'data/cifar-lenet_easy_first';
opts.train.expDir = opts.expDir;
ensuredir(opts.expDir);
%training order from easy to hard
net = cnn_cifar_init(opts) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.trainingOrder = 'fixed';
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;


%% training order from hard to easy
imdb = load(opts.imdbPath);
f_train = find(imdb.images.set==1);
f_test = find(imdb.images.set==3);

mean_perm = mean(perms_per_class);
[v,iv] = sort(mean_perm,'descend');
f_train = f_train(iv);
new_order = [f_train f_test];
imdb.images.data = imdb.images.data(:,:,:,new_order);
imdb.images.labels = imdb.images.labels(new_order);
imdb.images.set = imdb.images.set(new_order);

opts.expDir = 'data/cifar-lenet_hard_first';
opts.train.expDir = opts.expDir;
ensuredir(opts.expDir);
% First option : random training order.
net = cnn_cifar_init(opts) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.trainingOrder = 'fixed';
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3));



%% look-ahead training : at each point, select data maximizing the loss.
imdb = load(opts.imdbPath);

opts.expDir = 'data/cifar-lenet_adaptive_max_loss';
opts.train.expDir = opts.expDir;
ensuredir(opts.expDir);
% First option : random training order.
net = cnn_cifar_init(opts) ;
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.trainingOrder = 'adaptive_max_loss';
[net, info] = cnn_train_new(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% do some benchmarking
% [recalls,precisions,F_scores]  = doEvaluation_simple( opts,imdb,1,30,'normal');



