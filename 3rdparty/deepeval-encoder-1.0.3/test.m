% Copyright Ken Chatfield and Karen Simonyan, 2014
%
% this script gives a usage example of computing ConvNet features using
% the supplied MEX file
clear;

%TODO: Insert the path to an image here!
sample_image_path = 'sample/sample_image.jpg';

% REQUIRED model directory location
% model_dir = './models/CNN_M';
% model_dir = './models/CNN_M_2048';
% model_dir = './models/CNN_M_1024';
model_dir = './models/CNN_M_128';
% model_dir = './models/CNN_F';
% model_dir = './models/CNN_S';

fprintf('%s\n', model_dir);

param_file = sprintf('%s/param.prototxt', model_dir);
model_file = sprintf('%s/model', model_dir);

average_image = './models/mean.mat';

% optional - set whether to use GPU or CPU backend (default = CPU backend)
% this is done using a static class method of ConvNetEncoder, and is a
% global setting
% N.B. must be using a GPU-capable MEX file to allow GPU computation -
% see README file for details
use_gpu = false;

if use_gpu
    featpipem.directencode.ConvNetEncoder.set_backend('cuda');
    % can optionally set gpu device id using the second parameter:
    % featpipem.directencode.ConvNetEncoder.set_backend('cuda', 0);
end

% initialize an instance of the ConvNet feature encoder class
encoder = featpipem.directencode.ConvNetEncoder(param_file, model_file, ...
                                                average_image, ...
                                                'output_blob_name', 'fc7');

% use the encoder to compute the encoding from a single square crop from the
% centre of an image (the default if custom augmentation settings are not specified)
im = imread(sample_image_path);
im = featpipem.utility.standardizeImage(im); % ensure of type single, w. three channels

tic;
code = encoder.encode(im);
toc;

% recompute code for the image, this time enabling augmentation
%
% possible settings:
%  AUGMENTATION =
%    'none' *default* (no augmentation - use 224x224 crop from the image, rescaled so that the smallest side is 224)
%    'aspect_corners' (four corners + centre crop, and their flips, sampled from the image, rescaled so that the smallest side is 256)
%  AUGMENTATION_COLLATE =
%    'none' *default* (no collation - return additional crops as extra features)
%    'sum' (sum pooling)
%    'max' (max pooling)

encoder.augmentation = 'aspect_corners';
encoder.augmentation_collate = 'none';

% optionally set the backend to CUDA for the second computation
%featpipem.directencode.ConvNetEncoder.set_backend('cuda');

tic;
code_augmented = encoder.encode(im);
toc;

% it is also possible to instantiate multiple instances of the ConvNetEncoder
% class with different models, thus allowing the computation of features
% using multiple models simultaneously
model_dir_2 = './models/CNN_M';

param_file_2 = sprintf('%s/param.prototxt', model_dir_2);
model_file_2 = sprintf('%s/model', model_dir_2);

encoder_2 = featpipem.directencode.ConvNetEncoder(param_file_2, model_file_2, ...
                                                  average_image); % by default
                                                                  % fc7 is returned
tic;
code_2 = encoder_2.encode(im);
toc;
