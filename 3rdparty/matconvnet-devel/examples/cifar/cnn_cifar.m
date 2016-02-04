function [net, info] = cnn_cifar(varargin)
% CNN_CIFAR   Demonstrates MatConvNet on CIFAR-10
%    The demo includes two standard model: LeNet and Network in
%    Network (NIN). Use the 'modelType' option to choose one.

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
opts.modelType = 'lenet' ;
[opts, varargin] = vl_argparse(opts, varargin) ;
addpath(genpath('~/code/utils'));
opts.expDir = fullfile(vl_rootnn, 'data', ...
  sprintf('cifar-%s', opts.modelType)) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data','cifar') ;
opts.dataDir = '/net/mraid11/export/data/amirro/fcn/data/cifar_orig';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.whitenData = true ;
opts.contrastNormalization = true ;
opts.networkType = 'simplenn' ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

switch opts.modelType
  case 'lenet'
    net = cnn_cifar_init('networkType', opts.networkType) ;
  case 'nin'
    net = cnn_cifar_init_nin('networkType', opts.networkType) ;
  otherwise
    error('Unknown model type ''%s''.', opts.modelType) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn', trainfn = @cnn_train_dag ;
end
opts.train.gpus=2
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'val', find(imdb.images.set == 3)) ;


%%
startEpoch=1;
endEpoch=45;
nEpochs = net.meta.trainOpts.numEpochs;
% perfSuffix = '_lenet_train'
% [all_results,all_labels]  = getAllResults( opts,imdb,startEpoch,endEpoch,perfSuffix ,1 );
% [recalls,precisions,F_scores,per_class_losses,all_cms]  = doEvaluation( opts,imdb,startEpoch,endEpoch,perfSuffix,3 );
% 
perfSuffix = '_lenet_train';
[all_results,all_labels]  = getAllResults( opts,imdb,startEpoch,endEpoch,perfSuffix ,1 );
perf = zeros(nEpochs,1);
F_scores_train = zeros(nClasses,nEpochs);

for t = 1:length(perf)
    [losses1,cm1,precision1,recall1,F_score1] = evaluationMetrics(all_results{t},all_labels{t},10);
    F_scores_train(:,t) = F_score1;
    perf(t) = sum(diag(cm1))/sum(cm1(:));
%     clf; imagesc2(cm1); dpc(.2)
end

% find for each class it's maximal F_score.
[m,im] = max(F_scores_train,[],2);
figure(5);hist(im,1:45);

% make a score ensemble
% 1. without normalization
nClasses = 10;

all_results_softmaxed = {};
for t = 1:length(all_results)
    curRes = reshape(all_results{t}, 1,1,nClasses,[]);
    all_results_softmaxed{t} = squeeze(vl_nnsoftmax(curRes));
end
%%

%% no normalization
fprintf('trying to combine non-softmaxed results...\n');
newScores = zeros(size(all_results{1}));
for t = 1:nClasses            
    curBest = im(t);
    newScores(t,:) = all_results{curBest}(t,:);
end

% all(col(all_labels{t}==all_labels{t-1}))
[~,cm_new,precision_new,recall_new,F_score_new] = evaluationMetrics(newScores,all_labels{1},10);
perf_new = sum(diag(cm_new))/sum(cm_new(:));
fprintf('improvement: %%%2.2f\n',100*(perf_new-perf(end)));

%% 2. with normalization
fprintf('trying to combine softmaxed results...\n');
newScores = zeros(size(all_results{1}));
for t = 1:nClasses            
    curBest = im(t);
    newScores(t,:) = all_results_softmaxed{curBest}(t,:);
end

% all(col(all_labels{t}==all_labels{t-1}))
[~,cm_new,precision_new,recall_new,F_score_new] = evaluationMetrics(newScores,all_labels{1},10);
perf_new = sum(diag(cm_new))/sum(cm_new(:));
fprintf('improvement: %%%2.2f\n',100*(perf_new-perf(end)));

%% now do it on the test set. First,find out if the best performance is kept across test-train.
perfSuffix = '_lenet_test'

[all_results_test,all_labels_test]  = getAllResults( opts,imdb,startEpoch,endEpoch,perfSuffix ,3 );

perf_test = zeros(nEpochs,1);
F_scores_test = zeros(nClasses,nEpochs);
for t = 1:length(all_results_test)
    [losses1,cm1,precision1,recall1,F_score1] = evaluationMetrics(all_results_test{t},all_labels_test{t},10);
    F_scores_test(:,t) = F_score1;
    perf_test(t) = sum(diag(cm1))/sum(cm1(:));
%     clf; imagesc2(cm1); dpc(.2)
end

figure(6),plot(perf_test); title('test performance vs epoch');

% find for each class it's maximal F_score.
[m_test,im_test] = max(F_scores_test,[],2);
figure(7),plot(F_scores_test'); hold on;
for t = 1:length(im_test);
    plot(im_test(t),F_scores_test(t,im_test(t)),'r+','LineWidth',3);
end
title('location of best performance on test');
   
figure(5);hist(im_test,1:45);

% make a score ensemble
% 1. without normalization
nClasses = 10;

all_results_softmaxed_test = {};
for t = 1:length(all_results)
    curRes = reshape(all_results_test{t}, 1,1,nClasses,[]);
    all_results_softmaxed_test{t} = squeeze(vl_nnsoftmax(curRes));
end
%%

%% no normalization
fprintf('trying to combine non-softmaxed results...\n');
newScores_test = zeros(size(all_results_test{1}));
for t = 1:nClasses            
    curBest = im(t);
    newScores_test(t,:) = all_results_test{curBest}(t,:);
end
% all(col(all_labels{t}==all_labels{t-1}))
[~,cm_new,precision_new_test,recall_newtest,F_score_new_test] = evaluationMetrics(newScores_test,all_labels_test{1},10);
perf_new_test = sum(diag(cm_new))/sum(cm_new(:));
fprintf('improvement(test): %%%2.2f\n',100*(perf_new_test-perf_test(end)));

%% 2. with normalization
fprintf('trying to combine softmaxed results...\n');
newScores_test = zeros(size(all_results_test{1}));
for t = 1:nClasses            
    curBest = im(t);
    newScores_test(t,:) = all_results_softmaxed_test{curBest}(t,:);
end
% all(col(all_labels{t}==all_labels{t-1}))
[~,cm_new,precision_new_test,recall_newtest,F_score_new_test] = evaluationMetrics(newScores_test,all_labels_test{1},10);
perf_new_test = sum(diag(cm_new))/sum(cm_new(:));
fprintf('improvement(test): %%%2.2f\n',100*(perf_new_test-perf_test(end)));

%% 3. average last k results (w. softmax).
all_new_F_scores = zeros(size(F_scores_test));
for k = 1:45
    newScores_test = zeros(size(all_results_test{1}));
    
    for m = nEpochs-k+1:nEpochs
        newScores_test = newScores_test + all_results_softmaxed_test{m};
    end
   
    % all(col(all_labels{t}==all_labels{t-1}))
    [~,cm_new,precision_new_test,recall_newtest,F_score_new_test] = evaluationMetrics(newScores_test,all_labels_test{1},10);
    all_new_F_scores(:,k) = F_score_new_test;
    perf_new_test = sum(diag(cm_new))/sum(cm_new(:));
    fprintf('improvement(test, avg of last %d): %%%2.2f\n',k,100*(perf_new_test-perf_test(end)));
end

all_new_F_scores(:,end)-F_scores_test(:,end);

%% 

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% % -------------------------------------------------------------------------
% function [images, labels] = getSimpleNNBatch(imdb, batch)
% % -------------------------------------------------------------------------
% images = imdb.images.data(:,:,:,batch) ;
% labels = imdb.images.labels(1,batch) ;
% if rand > 0.5, images=fliplr(images) ; end
% 
% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if rand > 0.5, images=fliplr(images) ; end
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
%unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
unpackPath = fullfile(opts.dataDir);%, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean in any case
dataMean = mean(data(:,:,:,set == 1), 4);
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
