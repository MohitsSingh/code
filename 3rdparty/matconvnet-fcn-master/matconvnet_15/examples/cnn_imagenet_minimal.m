function cnn_imagenet_minimal()
% CNN_IMAGENET_MINIMAL   Minimalistic demonstration of how to run an ImageNet CNN model

% setup toolbox
run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matlab', 'vl_setupnn.m')) ;

% download a pre-trained CNN from the web
% if ~exist('imagenet-vgg-f.mat', 'file')
%   urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat', ...
%     'imagenet-vgg-f.mat') ;
% end
net = load('/home/amirro/storage/matconv_data/imagenet-vgg-f.mat') ;

% obtain and preprocess an image
im = imread('peppers.png') ;
im_ = single(im) ; % note: 255 range
im_ = imresize(im_, net.normalization.imageSize(1:2)) ;
im_ = im_ - net.normalization.averageImage ;

% run the CNN
net = vl_simplenn_move(net, 'gpu') ;
im_ = gpuArray(im_);
%%
im1 = repmat(im_,1,1,1,600);
profile off
% profile on
tt = tic;
opts.disableDropout = true;
opts.cudnn = false;
res = vl_simplenn(net, im_,[],[],opts) ;
tt1 = toc(tt);
tt1

% profile viewer

%%
XX = rand(27,27,256,256,'double');
%%
% clear X
X = gpuArray(XX);
%
tic
y = max(X,0);
toc
%%
% show the classification result
scores = squeeze(gather(res(end).x)) ;
[bestScore, best] = max(scores) ;
figure(1) ; clf ; imagesc(im) ;
title(sprintf('%s (%d), score %.3f',...
   net.classes.description{best}, best, bestScore)) ;

