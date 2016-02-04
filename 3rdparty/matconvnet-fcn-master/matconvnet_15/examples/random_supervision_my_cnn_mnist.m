% function [net, info] = mcnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

% run(fullfile(fileparts(mfilename('fullpath')), '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','mnist-baseline') ;
opts.train.expDir = opts.expDir ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBnorm = false ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 30 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.001 ;
% opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------
if exist(opts.imdbPath, 'file')
    imdb = load(opts.imdbPath) ;
else
    imdb = getMnistImdb(opts) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb') ;
end
opts.train.gpus=1;
net = cnn_mnist_init('useBnorm', opts.useBnorm) ;


[net, info] = cnn_train(net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;


net_from_random = trainRandomMnistArchitecture(100);

% visualize some of the first level filters.

% net_from_random.layers{1}.weights{1}

tt = 0;
for tt = 1:20
    subplot(4,5,tt);
    imagesc(net_from_random.layers{1}.weights{1}(:,:,:,tt))
end

new_net = net_from_random;
new_net.layers = new_net.layers(1:end-2);
f=1/100 ;
new_net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{f*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
new_net.layers{end+1} = struct('type', 'softmaxloss') ;
%% train the new net....
opts.train.errorFunction = 'multiclass';
opts.train.expDir='data/mnist-learn-init-from-random_100'
opts.train.numEpochs = 30;
[new_net, info] = cnn_train(new_net, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;



%% didn't work well, lets try a smaller number of networks. Start with one
net_15=trainRandomMnistArchitecture(15);
tt = 0;
for tt = 1:20
    subplot(4,5,tt);
    imagesc(net_2.layers{1}.weights{1}(:,:,:,tt))
end
net_15_bkp = net_15;
%% train the new net - again.
opts.train.errorFunction = 'multiclass';
opts.train.expDir='data/mnist-learn-init-from-random_15';
new_net15 = net_15_bkp;
new_net15.layers = new_net15.layers(1:end-2);
f=1/100 ;
new_net15.layers{end+1} = struct('type', 'conv', ...
    'weights', {{f*randn(1,1,500,10, 'single'), zeros(1,10,'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
new_net15.layers{end+1} = struct('type', 'softmaxloss') ;

[new_net15, info_15] = cnn_train(new_net15, imdb, @getBatch_simple_single_label, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;


%% load the original net...
A = load('data/mnist-baseline/net-epoch-20.mat');
net_baseline = A.net;
info_baseline = A.info;

figure(1),clf;
% plot(info_15.val.error(1,:),'r-');
hold on ,plot(info_15.val.error(2,:),'r--');
% plot(info_orig.val.error(1,:),'b-');
hold on ,plot(info_baseline.val.error(2,:),'b--');


% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
test_labels = imdb.images.labels(imdb.images.set==3);

aps_baseline = calcPerf(net_baseline,test_images,test_labels);
aps_rand_15 = calcPerf(new_net15,test_images,test_labels);

% figure,plot(aps_baseline,'r+'); hold on; plot(aps_rand_15,'g+')

new_net15.layers = new_net15.layers(1:end-1);
test_res = vl_simplenn(new_net15,test_images);
X = squeeze(test_res(end).x);


size(X)
aps = zeros(10,1);
for t = 1:10
    [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
    aps(t) = info.ap;
end
mean(aps)



% ok, now learn only the first label.
opts.expDir = fullfile('data','mnist-baseline-subset') ;
opts.train.expDir = opts.expDir ;

myopts = struct('useBnorm',opts.useBnorm,'subset',2);
net_partial = cnn_mnist_init(myopts);
imdb.images.labels(imdb.images.labels~=1) = 2;
[net_partial, info] = cnn_train(net_partial, imdb, @getBatch_simple, ...
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

