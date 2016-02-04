% function cnn_imagenet_minimal()
% CNN_IMAGENET_MINIMAL   Minimalistic demonstration of how to run an ImageNet CNN model

% setup toolbox
% run(fullfile(fileparts(mfilename('fullpath')), ...
%   '..', 'matlab', 'vl_setupnn.m')) ;
% 
% download a pre-trained CNN from the web
% if ~exist('imagenet-vgg-f.mat', 'file')
%   urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%     'imagenet-vgg-f.mat') ;
% end
net = load('/home/amirro/storage/matconv_data/imagenet-vgg-f.mat') ;
net.layers{end+1} = struct('type', 'softmaxloss', 'name', 'loss') ;

% obtain and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
net.layers{end}.class = 946;
res = vl_simplenn(net, im_) ;

% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;

opts.modelType = 'vgg-f' ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = false ;
opts.weightInitMethod = 'gaussian' ;                                  
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.expDir = '/home/amirro/storage/fcn/fool';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train.batchSize = 256 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = false ;
opts.train.gpus = [] ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = .001;
opts.train.numEpochs = 10;

imdb = struct;
qq=2;
% im_ = im_(qq:end-qq,qq:end-qq,:);
%%
imdb.images.data = {im_,im_};
imdb.images.label = [946 946]
imdb.images.set = [1 2];
fn = @(imdb,batch) my_simple_get_batch(imdb,batch);

for t = 1:length(net.layers)
    net.layers{t}.learningRate = [0 0];
end
opts.scale = 1;
firstLayer = struct('type', 'conv', 'name', sprintf('%s%s', 'img_changer','0'), ...
                           'weights', {{init_weight(opts, 3, 3, 3, 3, 'single'), zeros(3, 1, 'single')}}, ...
                           'stride', 1, ...
                           'pad', [16 16 16 16 ], ...
                           'learningRate', [1 2], ...
                           'weightDecay', [1 0]) ;
                       
newNet = net;
newNet.layers = {firstLayer, net.layers{:}};






% -------------------------------------------------------------------------
% now do a backprop on the scores to change the score into something else.
opts.train.skipSaving = true;

[net,info] = cnn_train(newNet, imdb, fn, opts.train, 'conserveMemory', true) ;

vl_simplenn

% now let's see what happens to the input image....
I_input = imdb.images.data{1};

res1 = vl_simplenn(net, I_input) ;

U = res1(2).x;
U = U-min(U(:));
U = U./max(U(:));

% show the filters
net.layers{1}.weights{1}
vl_imarraysc(net.layers{1}.weights{1})

