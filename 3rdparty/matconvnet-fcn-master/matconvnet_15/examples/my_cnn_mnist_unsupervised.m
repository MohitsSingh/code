% function [net, info] = mcnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');

opts.expDir = fullfile('data','mnist-unsupervised') ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBnorm = false ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = [] ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
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
net.layers{end+1} = struct('type','entropyloss');
opts.expDir = 'data/mnist-pretraining';

% create artificial data. This will be done as follows:
% 1. generate polygons of several types.
% 2. add a transformation and some noise.
%%
% nClasses = 3;
% N = 50000;
% R = randi(nClasses,1,N);
%%




opts.train.numEpochs=10;
opts.train.learningRate = 0.1;
opts.train.gpus = 1;
opts.train.batchSize = 10;
[net, info] = cnn_train(net, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
%%


%%

N = 50;
myData = zeros(2,2*N,'single');
myData(:,1:N) = randn(2,N)*5+20;
myData(:,N+1:end) = randn(2,N)*5+40;
myData = bsxfun(@minus,myData,mean(myData,2));
imdb2.images.data = reshape(myData,1,1,2,[]);
imdb2.images.labels = single(ones(1,2*N));
imdb2.images.labels(N+1:end) = 2;
sets = single(ones(1,2*N));
sets(1:2:end) = 3;
imdb2.images.set = sets;

f=1/100 ;
nHiddenUnits = 50;
net.layers = {} ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{f*randn(1,1,2,nHiddenUnits, 'single'), zeros(1, nHiddenUnits, 'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;
net.layers{end+1} = struct('type', 'relu') ;
% net.layers{end+1} = struct('type', 'conv', ...
%     'weights', {{f*randn(1,1,20,20, 'single'), zeros(1, 20, 'single')}}, ...
%     'stride', 1, ...
%     'pad', 0) ;
% net.layers{end+1} = struct('type', 'relu') ;
net.layers{end+1} = struct('type', 'conv', ...
    'weights', {{f*randn(1,1,nHiddenUnits,2, 'single'), zeros(1, 2, 'single')}}, ...
    'stride', 1, ...
    'pad', 0) ;

net.layers{end+1} = struct('type', 'softmax') ;
net.layers{end+1} = struct('type', 'entropyloss') ;
%%

% net.layers{end+1} = struct('type', 'softmaxloss') ;
opts.train.continue = false;
opts.train.gpus = 1;
opts.train.numEpochs = 10;
[net, info] = cnn_train(net, imdb2, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb2.images.set == 3)) ;

net2 = net;
net2.layers = net2.layers(1:end-1);
R = vl_simplenn(net2,imdb2.images.data(:,:,:,2:2:end));
[v,iv] = max(squeeze(R(end).x),[],1);
zz = squeeze(imdb2.images.data(:,:,:,2:2:end));
figure(1); clf; plot(zz(1,iv==1),zz(2,iv==1),'ro');
hold on; plot(zz(1,iv==2),zz(2,iv==2),'bd');


