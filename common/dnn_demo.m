
% download code and preferred models from here: http://www.vlfeat.org/matconvnet/

% add necessary paths
matConvNetDir = 'X:/code/3rdparty/matconvnet-1.0-beta7'; % main directory

% vlfeat required as well
addpath('X:/code/3rdparty/vlfeat-0.9.18/toolbox');
vl_setup

addpath(matConvNetDir);
addpath(fullfile(matConvNetDir,'examples'));
addpath(fullfile(matConvNetDir,'matlab'));

% initialize network. This needs to be done only once!!
% specify here the path to the model file downloaded above
net = init_nn_network('X:\code\3rdparty\matconvnet-1.0-beta7\imagenet-vgg-m-2048');

% prepare a batch of images for computation, features can be computed for each image independently but 
% it is faster in batches of e.g, 256 (the default batch size) images at a time
ims = {};
ims{1} = imread('D:\datasets\TUHOI\Images\ILSVRC2012_val_00000374.JPEG');
ims{2} = imread('D:\datasets\TUHOI\Images\ILSVRC2012_val_00000479.JPEG');

% extract last two fully connected layers of neural net as features.
% this produces 1 column vectors per layer per image. the names of the
% variables are due to the location of the layers in the net.

% not that many parameters are set inside this function , look in the
% examples of the original framework for tuning parameters
[x_17,x_19] = extractDNNFeats(ims,net);

% note: L2 normalization of features will probably lead to better
% classification results.

