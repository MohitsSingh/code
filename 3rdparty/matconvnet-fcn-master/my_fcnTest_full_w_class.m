% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath('utils/');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;
% opts.modelType = 'fcn32s' ;

opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-fcnTrain_action_region_full_w_class/net-epoch-33.mat';

opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat

load /net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/imdb.mat
opts.useGpu = 1;
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
gpuDevice(1)
[net_full_image,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);
net_full_image.move('gpu');

%%
% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
% imdb.nClasses = 3;
% numGpus = 0 ;
% nClasses = imdb.nClasses+1;
% confusion = zeros(nClasses) ;
%%
imgDir = '/home/amirro/storage/data/Stanford40//JPEGImages/';
preds = {};
scores = {};
load ~/code/mircs/fra_db_2015_10_08.mat
%%
% labels = {'bkg','obj','face','hand'};
labels_full = {'none','drink','smoke','bubbles','brush','phone'};

% labels = {'none','obj'};%,'','','','','',''};
% acfTrain
nLabels = length(labels_full);
confusion = zeros(nLabels);

for i=1:1215%i = 1:30:numel(test)
    i
    %imId = test(i) ;
    imId = i
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
    rgb = imResample(rgb,[384 384],'bilinear');
    [pred,scores,t] = predict_and_show(net_full_image,imageNeedsToBeMultiple,inputVar,rgb,1,true,labels_full);
    dpc
end

%     scores_hires_lm = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch_lm);
%     [~,preds_full_res_lm] = max(scores_hires_lm,[],3);

% %     showPredictions(rgb_orig,pred_full,softmaxScores_full,labels_full,1);
% %     showPredictions(rgb_orig,preds_full_res,scores_hires,labels_local,2);

%     showPredictions(rgb_orig,preds_full_res_lm,scores_hires_lm,labels_local_lm,3)

%     disp([tsmall t_medium t_big t_sub])

% %     dpc

%     ok = lb>0;

