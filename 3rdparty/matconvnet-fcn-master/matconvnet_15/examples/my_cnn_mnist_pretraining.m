% function [net, info] = mcnn_mnist(varargin)
% CNN_MNIST  Demonstrated MatConNet on MNIST

run(fullfile(fileparts(mfilename('fullpath')),...
  '..', 'matlab', 'vl_setupnn.m')) ;

addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
rmpath('/home/amirro/code/3rdparty/piotr_toolbox/external/other/');

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

opts.expDir = 'data/mnist-pretraining';

% create artificial data. This will be done as follows:
% 1. generate polygons of several types.
% 2. add a transformation and some noise.
%%
% nClasses = 3;
% N = 50000;
% R = randi(nClasses,1,N);
%%
% create base shapes
baseShapes = {};
f = maskCircle( 0, 2*pi, 10, 20);
baseShapes{end+1} = padarray(f,[4 4],0,'both');
f = maskCircle( -pi/8, pi/4, 10, 20 );
baseShapes{end+1} = padarray(f,[4 4],0,'both');
baseShapes{end+1} = baseShapes{end-1}.*(baseShapes{end}==0);
z = zeros(28);
z(4:23,4:23)=1;
baseShapes{end+1}=z;
z = zeros(28);
z(4:23,4:23)=1;
baseShapes{end+1} = z.*((imrotate(z,45,'bilinear','crop'))>0);
baseShapes{end+1} = z.*((imrotate(z,45,'bilinear','crop'))==0);

baseShapes{end+1} = baseShapes{4}.*(imerode(baseShapes{1},ones(3))==0);
% x2(baseShapes)


% jitter all the images!!

% how many examples to create from each image? for now, 5000.
%%
all_labels = {};
data = {};
for t = 1:length(baseShapes)
    t
    IJ = jitterImage(baseShapes{t},'nTrn',8,'mTrn',4,'mPhi',30,'nPhi',60); 
    IJ = IJ+randn(size(IJ))/10;
    all_labels{end+1} = ones(1,size(IJ,3))*t;
    data{end+1} = IJ;
end

all_labels = [all_labels{:}];
data = cat(3,data{:});
data = single(data);
dataMean = imdb.images.data_mean;
data = bsxfun(@minus, 255*data, dataMean) ;

imdb2 = imdb;
imdb2.images.data = reshape(data,28,28,1,[]);
imdb2.images.set = ones(1,size(imdb2.images.data,4));
imdb2.images.set(1:5:end) = 3;
imdb2.images.labels = all_labels;
% imdb2.images.data = imdb2.images.data*255;
%%
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
opts.train.expDir = '/data/mnist-pretraining';
opts.train.expDir = opts.expDir;
opts.train.batchSize = 100;
opts.train.learningRate = 0.001;
opts.train.numEpochs=10;
[net, info] = cnn_train(net, imdb2, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb2.images.set == 3)) ;

%% initialize....

% now train on the original data.
opts.train.expDir = 'data/mnist-after_pretraining';
imdb = load(opts.imdbPath) ;
opts.train.numEpochs=10;
opts.train.learningRate = 0.001;
[net, info] = cnn_train(net, imdb, @getBatch_simple, ...
    opts.train, ...
    'val', find(imdb.images.set == 3)) ;




imgs = imdb.images.data(:,:,:,1:1000);

vecs = reshape(imgs,[],1000);

[ U, mu, vars ] = pca( vecs);



v1 = U*vecs(:,1);

U*v1


% create training data using SVD;


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








