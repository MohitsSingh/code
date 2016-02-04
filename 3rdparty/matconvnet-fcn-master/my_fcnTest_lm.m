% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath utils
% experiment and data paths


opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;
% opts.modelType = 'fcn32s' ;


% opts.expDir = 'data/fcn32s-voc11' ;
% opts.dataDir = 'data/voc11' ;
%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn32s-action_obj/net-epoch-50.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_8/net-epoch-100.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_w_hands/net-epoch-100.mat';
opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_landmarks/net-epoch-46.mat';
opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_8_x2_landmarks_and_objects/net-epoch-100.mat';
%%
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;

% experiment setup

load ~/storage/misc/images_and_masks_x2_w_hands.mat
train= find(isTrain);
val = train(1:3:end);
train = setdiff(train,val);
imdb.images = 1:length(images);
imdb.images_data = images;
imdb.labels = landmarks;
imdb.meta.classes  = {'none','eye','mouth center','mouth corner','chin center','nose tip'};   
needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
if needToConvert
    for t = 1:length(images)
        imdb.images_data{t} = im2uint8(imdb.images_data{t});
    end
end
%   imdb.labels = masks;
[net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

numGpus = 0 ;
imdb.meta.classes  = {'none','eye','mouth ','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};   
nClasses = length(imdb.meta.classes);
confusion = zeros(nClasses) ;
net.move('gpu');


%%
%test =setdiff(1:1215,union(train,val));
test = setdiff(1:1215,union(train,val));
% test = val

for i = 1:30:numel(test)
    imId = test(i) ;    
    rgb = imdb.images_data{imId};
    rgb = single(rgb);        
    sizes = [300 400 500 600];
%     sizes = [384];
    sz_orig = size2(rgb);
    rgb_orig = rgb;
    scores_ms = zeros([sz_orig nClasses]);
    for iSize = 1:length(sizes)        
        iSize
        s = sizes(iSize);
        rgb = imResample(rgb_orig,[s s],'bilinear');
        [~,scores_] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,rgb,1,false,imdb.meta.classes);
        softmaxScores = bsxfun(@rdivide,exp(scores_),sum(exp(scores_),3));
        scores_ms = scores_ms+imResample(softmaxScores,sz_orig,'bilinear');                
%         dpc
    end
    [~,pred] = max(scores_ms,[],3);
    showPredictions(rgb_orig,pred,scores_ms/length(sizes),imdb.meta.classes,1);
%     dpc
%     continue;
    
%     rgb = single(rgb);    
    rgb = imResample(rgb_orig,[384 384]);    
    [pred_full,scores_] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,rgb,1,false,imdb.meta.classes);    
    softmaxScores = bsxfun(@rdivide,exp(scores_),sum(exp(scores_),3));
    scores = imResample(softmaxScores,sz_orig,'bilinear');   
    [~,pred] = max(scores,[],3);
    showPredictions(rgb_orig,pred,scores,imdb.meta.classes,2);
%     sum(pred_full(:)==6)
    dpc;
end

