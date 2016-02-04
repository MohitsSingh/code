% configuration...

% general
conf.prefix = '~/storage/2011_test_largesp';
conf.VOCopts = VOCopts;

%
ensuredir(conf.prefix);

% on which image-set to operate?
conf.trainSet = 'train';
conf.testSet = 'val';
% conf.imageSet = 'trainval';

% use full segmentations only or only bounding boxes?
conf.mode = 'seg'; % change to 'bb' to use bounding boxes.
% conf.mode = 'bb';

% number of neighbors to choose of each type
conf.n_bow_neighbors = 30;
conf.n_gist_neighbors = 30;

% superpixel parameters
conf.superpixels.coarse_size = 100;
conf.superpixels.fine_size = 15;
conf.superpixels.coarse_regularization = .05;
conf.superpixels.fine_regularization = .05;

% load all quantized descriptors and feature locations at once.
% this is much more memory intensive but considerably faster
% set to false if have get out of memory issues,.
conf.preLoadFeatures = true;

% apply or not outlier removal in confidence maps
conf.removeProbOutliers = true;

% how much to filter shape neighbors?
conf.gaussian.hsize = 19;
conf.gaussian.hsigma = 7;

% gamma correction for shape prior
conf.shapeGamma = .25;

% weight of edges in graph
conf.graphEdgeWeight = 15;