% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath utils;

% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;
% opts.modelType = 'fcn32s' ;

% opts.expDir = 'data/fcn32s-voc11' ;
% opts.dataDir = 'data/voc11' ;
%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn32s-action_obj/net-epoch-50.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_8/net-epoch-100.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_w_hands/net-epoch-100.mat';
%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_landmarks_and_objects/net-epoch-100.mat';
opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj/net-epoch-50.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_landmarks/net-epoch-76.mat';
%%
opts.modelFamily = 'matconvnet' ;
% opts.modelFamily = 'ModelZoo' ;
opts.nClasses = 0;

load ~/storage/misc/images_and_masks_x2_w_hands.mat

train= find(isTrain);
test = find(~isTrain);
val = train(1:3:end);
train = setdiff(train,val);

imdb.images = 1:length(images);
imdb.images_data = images;
needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
if needToConvert
    for t = 1:length(images)
        imdb.images_data{t} = im2uint8(imdb.images_data{t});
    end
end
imdb.labels = masks;
imdb.nClasses = 3;

% opts = vl_argparse(opts, varargin) ;
opts.useGpu = 0;
% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------
% Get PASCAL VOC 11/12 segmentation dataset plus Berkeley's additional
% segmentations

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
%%
[net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath,opts.modelFamily);

% % -------------------------------------------------------------------------
% % Train
% -------------------------------------------------------------------------

imdb.nClasses = 1;
numGpus = 0 ;
nClasses = imdb.nClasses+1;
confusion = zeros(nClasses) ;
net.move('gpu');
load ~/code/mircs/fra_db_2015_10_08.mat
imgDir = '/home/amirro/storage/data/Stanford40//JPEGImages/';
%labels = {'none','face','hand','drink','smoke','blow','brush','phone'};
labels = {'none','obj'};%,'hand','drink','smoke','blow','brush','phone'};

%%
for i = 1:1:numel(test)
    i
    imId = test(i);    
%     rgb = single(imread(fullfile(imgDir,fra_db(imId).imageID)));
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
%     lb = imdb.labels{imId};    
    %rgb = imResample(rgb,[384 384],'bilinear');
    %rgb = imResample(rgb,768/size(rgb,1),'bilinear');
%     rgb = imResample(rgb,768/size(rgb,1),'bilinear');
%     lb = imResample(lb,size2(rgb),'nearest');    
    [pred_full,scores_full] = predict_and_show(net,imageNeedsToBeMultiple,inputVar_tvg,'coarse',rgb,1,true,labels);
    dpc
end
%%

%% run on all images.
load ~/storage/misc/images_detected.mat


imdb.images_data = images;
imdb = rmfield(imdb,'labels');
%%
results = {};




batches = batchify(length(imdb.images),150);
for t = 1:length(batches)
    t
    curBatch = batches{t};
    y = getBatch_action_obj(imdb, curBatch,'labelOffset',1,'useFlipping',false,'doScaling',false);
    rgb = y{2};
    im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;
    if imageNeedsToBeMultiple
        sz = [size(im,1), size(im,2)] ;
        sz_ = round(sz / 32)*32 ;
        im_ = imresize(im, sz_) ;
    else
        im_ = im ;
    end
    net.eval({inputVar, gpuArray(im_)}) ;
    scores_ = gather(net.vars(net.getVarIndex('prediction')).value);
    for u = 1:length(curBatch)
        results{curBatch(u)} = imResample(scores_(:,:,:,u),size2(imdb.images_data{curBatch(u)}));
    end
    %   clf; subplot(1,2,1); imagesc2(images{t});
    %   subplot(1,2,2); imagesc2(results{t});
    %   dpc
end
%%
save ~/code/mircs/images_and_face_obj_results.mat results
