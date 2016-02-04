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

opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-fcnTrain_action_region_full/net-epoch-28.mat';
% opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat

load /net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/imdb.mat
opts.useGpu = 1;
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
gpuDevice(2)
subPatchesPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_landmarks_and_objects/net-epoch-100.mat';

% subpatches - trained on the output of the full image detector
[net_sub_patch,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(subPatchesPath, opts.modelFamily);
net_sub_patch.move('gpu');

% net_sub_patch.delete()

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
labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};
labels_local_lm = {'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};
labels_local = labels_full;%{'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};
% labels = {'none','obj'};%,'','','','','',''};
% acfTrain
nLabels = length(labels_full);
confusion = zeros(nLabels);
outPath = '~/storage/fra_action_fcn_hires_with_lm';
mkdir(outPath);
p = randperm(1215);
for i=p%i = 1:30:numel(test)
    i
    %imId = test(i) ;
    imId = i
    curResPath = j2m(outPath,fra_db(i));
% % %     if (exist(curResPath,'file'))
% % %         
% % %         d = dir(curResPath);
% % %         if d.bytes > 0
% % %             continue          
% % %         end
% % %     end    
%     fclose(fopen(curResPath,'w+'));
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
    rgb_orig = rgb;
    sz_orig = size2(rgb);
%     rgb = imResample(rgb,[384 384],'bilinear');
    tic
    
    resizeRatio = 1152/size(rgb_orig,1);
   
    rgb_orig_x = imResample(rgb_orig,1152/size(rgb_orig,1),'bilinear');
    
    [scores_hires_full,pred] = applyNet(net_sub_patch,rgb_orig_x,imageNeedsToBeMultiple,inputVar);
%     [scores_hires_full,pred_hires_full,t_big] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,rgb_orig_x,1,false,labels_local);
%     [scores_hires_full,pred_hires_full,t_big] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,rgb_orig_x,1,true,labels_local);
    scores_hires_full = imResample(scores_hires_full,sz_orig,'bilinear');
    pred = imResample(pred,sz_orig,'nearest');
    softmaxScores = bsxfun(@rdivide,exp(scores_hires_full),sum(exp(scores_hires_full),3));
    showPredictions(rgb_orig,pred,softmaxScores,labels_local_lm,1)
    dpc; continue
    
    % % % % % % %     [x,y] = meshgrid(1:sz_orig(2),1:sz_orig(1));
    % % % % % % %     m = M(:);
    % % % % % % %     xy_centroid = sum([x(:).*m y(:).*m])/sum(m);
    % % % % % % %     bb = round(inflatebbox(xy_centroid,2*size(rgb_orig,1)/3,'both',true));
    % % % % % % %     rgb_orig_medium = imResample(cropper(rgb_orig,bb),resizeRatio,'bilinear');
    % % % % % % %     [~,~,t_medium] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,rgb_orig_medium,4,true,labels_local);
    % % % % % % %     % calc mean distance to center          
    save(curResPath,'scores_hires_full');
end

%     scores_hires_lm = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch_lm);
%     [~,preds_full_res_lm] = max(scores_hires_lm,[],3);

% %     showPredictions(rgb_orig,pred_full,softmaxScores_full,labels_full,1);
% %     showPredictions(rgb_orig,preds_full_res,scores_hires,labels_local,2);

%     showPredictions(rgb_orig,preds_full_res_lm,scores_hires_lm,labels_local_lm,3)

%     disp([tsmall t_medium t_big t_sub])

% %     dpc

%     ok = lb>0;

