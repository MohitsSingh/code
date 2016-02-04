% function [net, info] = my_cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

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

switch opts.modelType
    case 'lenet', net = cnn_cifar_init(opts) ;
    case 'nin',   net = cnn_cifar_init_nin(opts) ;
end

if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getCifarImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;



%%
% train on a single class
wanted_classes = 1;
other_class = 10;
imdb_sub = restrict_to_sub(imdb,wanted_classes,other_class);

switch opts.modelType
    case 'lenet', net = cnn_cifar_init(opts) ;
    case 'nin',   net = cnn_cifar_init_nin(opts) ;
end
opts.expDir = fullfile('data','cifar-baseline-subset') ;
opts.train.expDir = opts.expDir ;


[net, info] = cnn_train(net, imdb_sub, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

%%
% --------------------------------------------------------------------

% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
net.layers = net.layers(1:end-1);
test_res = vl_simplenn(net,test_images);
X = squeeze(test_res(end).x);
size(X)
test_labels = imdb.images.labels(imdb.images.set==3);
aps = zeros(10,1);
for t = 1:10
    [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
    aps(t) = info.ap;
end

%%


%%

% now make a new random network.
net_from_random = trainRandomCIFARArchitecture(50);
% ok, now learn only the first label.
opts.expDir = fullfile('data','cifar-baseline-subset') ;
opts.train.expDir = opts.expDir ;
myopts = struct('useBnorm',opts.useBnorm,'subset',2);
net_partial = cnn_cifar_init(myopts);
imdb.images.labels(imdb.images.labels~=1) = 2;
[net_partial, info] = cnn_train(net_partial, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% test this net on the first label
net_partial.layers = net_partial.layers(1:end-1);
test_res = vl_simplenn(net_partial,test_images);
X = squeeze(test_res(end).x);
size(X)
test_labels = imdb.images.labels(imdb.images.set==3);
aps = zeros(2,1);
for t = 1:2
    [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
    aps(t) = info.ap;
end

% --------------------------------------------------------------------

