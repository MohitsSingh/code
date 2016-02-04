% function [net, info] = mcnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...  
  '..', 'matlab', 'vl_setupnn.m')) ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath(genpath('~/code/utils'));
opts.expDir = fullfile('data','mnist-baseline') ;
% [opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data','mnist') ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.useBnorm = false ;
opts.train.batchSize = 100 ;
opts.train.numEpochs = 20 ;
opts.train.continue = true ;
opts.train.gpus = 1 ;
opts.train.learningRate = 0.001 ;
opts.train.expDir = opts.expDir ;
% opts = vl_argparse(opts, varargin) ;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------



%% try different initializations for the mnist data....

% 1. random data patches (sampled from high-energy locations)
% 2. Patch-PCA : pca on the patches
% 3. data-driven PCA : sampled which maximize the projected variance
% 4. gabor filters. 
gpuDevice(1);
orig_imdbPath = 'data/mnist-baseline/imdb.mat';
imdb = load(orig_imdbPath) ;
imdb.images.labels = reshape(imdb.images.labels,1,1,1,[]);
opts.train.gpus=1;
net = cnn_mnist_init('useBnorm', opts.useBnorm) ;

[net_orig, info] = cnn_train(net, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
% now check the mean average precision
net_orig.layers = net_orig.layers(1:end-1);
net_orig = vl_simplenn_move(net_orig, 'gpu') ;
test_images = imdb.images.data(:,:,:,imdb.images.set==3);            
test_labels = imdb.images.labels(imdb.images.set==3);
[cm_orig,X] = calcPerf(net_orig,test_images,test_labels,1);
mean(diag(cm_orig))

%%
net = cnn_mnist_init('useBnorm', opts.useBnorm) ;
w = 5;
% random patches
firstLayerWeights = net.layers{1}.weights{1};
opts.train.expDir ='data/mnist-random_patches';
ensuredir(opts.train.expDir);
% choose 20 random patches.
train_images = find(imdb.images.set==1);
tr = train_images(randperm(length(train_images)));
tr = tr(1:20);
tr = imdb.images.data(10:14,10:14,:,tr);
norms = sum(sum(tr.^2,1),2);
tr = bsxfun(@rdivide,tr,norms.^.5);
net.layers{1}.weights{1} = tr;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

[net_random_patches, info] = cnn_train(net_random_patches, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
% now check the mean average precision
net_random_patches.layers = net_random_patches.layers(1:end-1);
net_random_patches = vl_simplenn_move(net_random_patches, 'gpu') ;
test_images = imdb.images.data(:,:,:,imdb.images.set==3);            
test_labels = imdb.images.labels(imdb.images.set==3);
[cm_rp,X] = calcPerf(net_random_patches,test_images,test_labels,1);
mean(diag(cm))


%% do it again, only this time use pca on the patches.
tr = train_images(randperm(length(train_images)));
tr = tr(1:1000);
[y,x] = meshgrid(1:5:21,1:5:21);
tr = imdb.images.data(:,:,:,tr);
tr_ = {};
locs = randint2(1,size(tr,4),[1 length(x(:))]);
for t = 1:length(locs)
    m = locs(t);
    p = tr(x(m):x(m)+4,y(m):y(m)+4,:,t);
    tr_{t} =p;
end
tr_ = cat(4,tr_{:});

tr_ = reshape(tr_,[],size(tr_,4));
[ U, mu, vars ] = pca( tr_ );
U = reshape(U(:,1:20),5,5,1,[]);

opts.train.expDir ='data/mnist-pca_patches';
ensuredir(opts.train.expDir);
net_pca_patches = cnn_mnist_init('useBnorm', opts.useBnorm) ;
net_pca_patches.layers{1}.weights{1} = single(U);
[net_pca_patches, info] = cnn_train(net_pca_patches, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

net_pca_patches = vl_simplenn_move(net_pca_patches, 'gpu') ;  
net_pca_patches.layers = net_pca_patches.layers(1:end-1);
[cm_pca,X] = calcPerf(net_pca_patches,test_images,test_labels,1);
mean(diag(cm_pca))

%% now do the same but the most representative patches instead of calculated vectors:
% for each pca patch actually find the nearest neighboring vector.
tr = train_images(randperm(length(train_images)));
tr = tr(1:1000);
[y,x] = meshgrid(1:5:21,1:5:21);
tr = imdb.images.data(:,:,:,tr);
tr_ = {};
locs = randint2(1,size(tr,4),[1 length(x(:))]);
for t = 1:length(locs)
    m = locs(t);
    p = tr(x(m):x(m)+4,y(m):y(m)+4,:,t);
    tr_{t} =p;
end
tr_ = cat(4,tr_{:});
tr_ = reshape(tr_,[],size(tr_,4));
[ U, mu, vars ] = pca( tr_ );

tr_n = normalize_vec(tr_);

V = abs(tr_n'*U);
sel_ = zeros(20,1);
% [r,ir] = max(V,[],1);
for t = 1:20
    [r,ir] = max(V(:,t));
    sel_(t) = ir;
    V(ir,:) = 0;
end

tr_n = tr_n(:,sel_);
U = reshape(tr_n(:,1:20),5,5,1,[]);
opts.train.expDir ='data/mnist-pca_patches_sel';
ensuredir(opts.train.expDir);
net_pca_patches_sel = cnn_mnist_init('useBnorm', opts.useBnorm) ;
net_pca_patches_sel.layers{1}.weights{1} = single(U);

[net_pca_patches_sel, info] = cnn_train(net_pca_patches_sel, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;

net_pca_patches_sel = vl_simplenn_move(net_pca_patches_sel, 'gpu') ;  
net_pca_patches_sel.layers = net_pca_patches_sel.layers(1:end-1);
[cm_pca_sel,X] = calcPerf(net_pca_patches_sel,test_images,test_labels,1);
mean(diag(cm_pca_sel))
%%

all_cms = {};
test_images = imdb.images.data(:,:,:,imdb.images.set==3);            
test_labels = imdb.images.labels(imdb.images.set==3);
for ii = 1:23
        ii
    load(sprintf('data/mnist-baseline/net-epoch-%d.mat',ii));
    net = vl_simplenn_move(net, 'gpu') ;    
    net.layers = net.layers(1:end-1);        
%     test_images = test_images;
%     test_labels = test_labels(1:500);
    [cm,X] = calcPerf(net,test_images,test_labels,1);
%     test_images = imdb.images.data(:,:,:,imdb.images.set==3);
%     test_labels = imdb.images.labels(imdb.images.set==3);
    all_cms{ii} = cm;
end

diags = {};
for t = 1:length(all_cms)
    diags{t} = diag(all_cms{t}(1:10,1:10));
end
diags = cat(2,diags{:});

% plot(diags')
% 
% 
% all_aps_val = {};
% val
% for ii = 1:30
%         ii
%     load(sprintf('data/cifar-100-nin/net-epoch-%d.mat',ii));
%     net = vl_simplenn_move(net, 'gpu') ;    
%     [aps,X] = calcPerf(net,test_images,test_labels,10);
%     all_aps_val{ii} = aps;
% end

bestPerf = max(diags,[],2);
finalPerf = diags(:,end);
figure,plot(finalPerf,bestPerf,'r+'); xlabel('final'); ylabel('best');
figure,plot(max(diags,[],2)./diags(:,end))
% figure,plot(diags(96,:));


% Interesting, the best attained performance can be much better than the
% final one...
mean(diags)
figure(); clf; hold on; plot(diags(:,end),'g-');
plot(max(diags,[],2),'b-');
figure,stem(bestPerf-finalPerf)
mean(bestPerf-finalPerf)
%%

% test_res = vl_simplenn(net,test_images);
% X = squeeze(test_res(end).x);
% size(X)
% test_labels = imdb.images.labels(imdb.images.set==3);
% aps = zeros(10,1);
% for t = 1:10
%     [prec,rec,info] = vl_pr(2*(test_labels==t)-1,X(t,:));
%     aps(t) = info.ap;
% end

% ok, now learn only the first label.
opts.expDir = fullfile('data','mnist-baseline-subset') ;
opts.train.expDir = opts.expDir ;
myopts = struct('useBnorm',opts.useBnorm,'subset',2);
new_new = cnn_mnist_init(myopts);
imdb.images.labels(imdb.images.labels~=1) = 2;


% now do class-augmentation, for the numbers:
%2,3,4,5,6,7,9 (1 is sort of symmetric)
labels = imdb.images.labels;

new_imgs = {};
new_labels = {};
new_sets = {};

classes_to_flip = [2,3,4,5,6,7,9];
for iClass = 1:length(classes_to_flip)
    curClass = classes_to_flip(iClass);
    curImgs = imdb.images.data(:,:,:,labels==curClass);
    curImgs = flip(curImgs,2);
    new_imgs{iClass} = curImgs;
    new_labels{iClass} = ones(1,size(curImgs,4))*(10+iClass);    
end
new_imgs = cat(4,new_imgs{:});
new_labels = cat(2,new_labels{:});
new_sets = ones(size(new_labels));
new_sets(1:6:end) = 3;

imdb_new = imdb;

imdb_new.images.data = cat(4,imdb.images.data,new_imgs);
imdb_new.images.labels = cat(2,imdb.images.labels,new_labels);
imdb_new.images.set = cat(2,imdb.images.set,new_sets);
imdb_new.images.labels = reshape(imdb_new.images.labels,1,1,1,[]);
opts.expDir = fullfile('data','mnist-w_flipped_samples') ;
opts.train.expDir = opts.expDir ;

net_new = cnn_mnist_init('useBnorm', opts.useBnorm) ;
% net = cnn_mnist_init('useBnorm', opts.useBnorm) ;

[net_new, info] = cnn_train(net_new, imdb_new, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb_new.images.set == 3)) ;


test_images = imdb.images.data(:,:,:,imdb.images.set==3);
test_labels = imdb.images.labels(imdb.images.set==3);
[aps_new,X_new] = calcPerf(net_new,test_images,test_labels);





