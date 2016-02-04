% function [net, info] = my_cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

% run(fullfile(fileparts(mfilename('fullpath')), ...
% %   '..', 'matlab', 'vl_setupnn.m')) ;

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

[net, info] = cnn_train(net, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% --------------------------------------------------------------------

% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
% net.layers = net.layers(1:end-1);
% test_res = vl_simplenn(net,test_images);
% X = squeeze(test_res(end).x);
% size(X)
% test_labels = imdb.images.labels(imdb.images.set==3);
% aps = zeros(10,1);
% for t = 1:10
%     [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
%     aps(t) = info.ap;
% end

% now make a new random network.


K= 100;
net_from_random = trainRandomCIFARArchitecture(K);
clf;
tt = 0;
for tt = 1:32
    tt
    subplot(5,7,tt);
    imagesc(normalise(squeeze(net_from_random.layers{1}.weights{1}(:,:,:,tt))));
end


%%
%% train the new net - again.
opts.train.errorFunction = 'multiclass';
opts.train.expDir=['data/cifar-learn-init-from-random_' num2str(K)];
new_net = net_from_random;
% opts.train.learningRate = opts.train.learningRate/100;
new_net.layers = new_net.layers(1:end-1);
lr = [.1 2] ;
new_net.layers{end+1} = struct('type', 'conv', ...
                           'weights', {{0.05*randn(1,1,K,10, 'single'), zeros(1,10,'single')}}, ...
                           'learningRate', .1*lr, ...
                           'stride', 1, ...
                           'pad', 0) ;

new_net.layers{end+1} = struct('type', 'softmaxloss') ;

[new_net, new_info] = cnn_train(new_net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
test_labels = imdb.images.labels(imdb.images.set==3);

[aps_baseline,X_baseline] = calcPerf(net,test_images,test_labels);
[aps_rand,X_rand] = calcPerf(new_net,test_images,test_labels);
% [aps_rand,X_rand] = calcPerf(new_net,test_images(:,:,:,1:100),test_labels(1:100));

% figure,plot(aps_baseline,'r+'); hold on; plot(aps_rand_50,'g+')


%%

% ok, now learn only the first label.
% opts.expDir = fullfile('data','cifar-baseline-subset') ;
opts.train.expDir = opts.expDir ;
% myopts = struct('useBnorm',opts.useBnorm,'subset',2);
net_partial = cnn_cifar_init();
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

