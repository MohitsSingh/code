% function [net, info] = mcnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

opts.expDir = fullfile('data','mnist-baseline') ;
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

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.batchSize = 256;
opts.train.learningRate = .001;
[net, info] = cnn_train(net, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;
% now check the mean average precision
test_images = imdb.images.data(:,:,:,imdb.images.set==3);
net.layers = net.layers(1:end-1);



net = vl_simplenn_move(net, 'gpu') ;



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








