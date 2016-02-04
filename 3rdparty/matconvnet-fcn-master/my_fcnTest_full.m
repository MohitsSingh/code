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

%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-fcnTrain_action_objects_8_single_obj_full/net-epoch-94.mat';

opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_only_full/net-epoch-39.mat';


%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-fcnTrain_action_region_full/net-epoch-28.mat';
% opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
% opts.modelPath  = '/net/mraid11/export/data/amirro/fcn/data/fcn8s_person_parts_focused/net-epoch-33.mat';
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat

load /net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_classes_full/imdb.mat

opts.useGpu = 1;
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
% gpuDevice(1)
[net_full_image,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);
net_full_image.move('gpu');

%
% landmark and object
% % % % subPatchesPath_w_landmarks = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_landmarks_and_objects/net-epoch-100.mat';
% % % % [net_sub_patch_lm,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(subPatchesPath_w_landmarks, opts.modelFamily);
% % % % % just classes, zoomed in
% % % % 
% % % % net_sub_patch_lm.move('gpu');
% % % % subPatchesPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes/net-epoch-100.mat';
% % % % 
% % % % % subpatches - trained on the output of the full image detector
% % % % % subPatchesPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-fcnTrain_action_objects_classes_subpatches/net-epoch-17.mat';
% % % % [net_sub_patch,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(subPatchesPath, opts.modelFamily);
% % % % net_sub_patch.move('gpu');

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

% load('/home/amirro/storage/mircs_18_11_2014/s40_fra.mat');
% fra_db = s40_fra;
%%
% labels = {'bkg','obj','face','hand'};
%labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};

labels_full = {'none','drink','smoke','bubbles','brush','phone'};
% labels_full = {'none','face','hand','object'}
% labels_local_lm = {'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};
% labels_local = labels_full;%{'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};
% labels_full = {'none','face','hand','object'};
% labels = {'none','obj'};%,'','','','','',''};
% acfTrain
nLabels = length(labels_full);
confusion = zeros(nLabels);
outPath = '~/storage/fra_action_fcn_hires';
% mkdir(outPath);
L =load('~/storage/misc/voc10_action_trainval.mat');
% test = randperm(length(L.action_gt_train));
% p = randperm(length(fra_db));
for i = 1:1:numel(train)
    i
    imId = train(i) ;
%     curImgData = L.action_gt_train(imId);    
%     imId = i
% % %     curResPath = j2m(outPath,fra_db(i));
% % %     if (exist(curResPath,'file'))
% % %         d = dir(curResPath);
% % %         if d.bytes > 0
% % %             continue
% % %         end
% % %     end    
% % %     fclose(fopen(curResPath,'w+'));
    rgb = single(imdb.images_data{imId});% ( {fullfile(imgDir,fra_db(imId).imageID)});
% %      imagePath = fullfile(L.vocImgDir,[curImgData.imageID '.jpg']);
% %      rgb = vl_imreadjpeg({imagePath});rgb = rgb{1};
%     rgb = vl_imreadjpeg({j2m(imgDir,fra_db(i),'.jpg')});rgb = rgb{1};
% %     curBox = makeSquare(curImgData.personBox,true);
% %     curBox = round(inflatebbox(curBox,1.2,'both',false));
    
% %     rgb = cropper(rgb,curBox);
    rgb_orig = rgb;
    sz_orig = size2(rgb);
%     rgb = imResample(rgb,[384 384],'bilinear');
    tic
    
    predict_and_show(net_full_image,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,true,labels_full);
    dpc
    continue
    
    [scores_full_image,pred_full] = applyNet(net_full_image,rgb,imageNeedsToBeMultiple,inputVar);
    tsmall = toc;
    % get subpatches and show prediction for each sub-patch
    scores_full_image = imResample(scores_full_image,sz_orig,'bilinear');
    pred_full = imResample(pred_full,sz_orig,'nearest');
    softmaxScores_full = bsxfun(@rdivide,exp(scores_full_image),sum(exp(scores_full_image),3));
    M = max(softmaxScores_full(:,:,2:end),[],3);
    % go over sub-windows and aggerate results.
    [subs,vals] = nonMaxSupr( double(M), 20, .1,10 );
    subs = fliplr(subs);
    boxes = round(inflatebbox(subs,mean(sz_orig)/3,'both',true));
    curPatches = multiCrop2(rgb_orig,boxes);
    %
    %     tic   
    % % % % % % %     resizeRatio = 1152/size(rgb_orig,1);
    % % % % % % %     [~,~,t_big] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,rgb_orig_x,3,false,labels_local);
    % % % % % % %     rgb_orig_x = imResample(rgb_orig,1152/size(rgb_orig,1),'bilinear');
    % % % % % % %     [x,y] = meshgrid(1:sz_orig(2),1:sz_orig(1));
    % % % % % % %     m = M(:);
    % % % % % % %     xy_centroid = sum([x(:).*m y(:).*m])/sum(m);
    % % % % % % %     bb = round(inflatebbox(xy_centroid,2*size(rgb_orig,1)/3,'both',true));
    % % % % % % %     rgb_orig_medium = imResample(cropper(rgb_orig,bb),resizeRatio,'bilinear');
    % % % % % % %     [~,~,t_medium] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,rgb_orig_medium,4,true,labels_local);
    % % % % % % %     % calc mean distance to center
    
    %     title(num2str(t0))
    [scores_hires,t_sub] = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch);
    [scores_hires_lm,t_sub] = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch);
    [~,preds_full_res] = max(scores_hires,[],3);    
    save(curResPath,'scores_full_image','boxes','scores_hires','scores_hires_lm');
end

%     scores_hires_lm = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch_lm);
%     [~,preds_full_res_lm] = max(scores_hires_lm,[],3);

% %     showPredictions(rgb_orig,pred_full,softmaxScores_full,labels_full,1);
% %     showPredictions(rgb_orig,preds_full_res,scores_hires,labels_local,2);

%     showPredictions(rgb_orig,preds_full_res_lm,scores_hires_lm,labels_local_lm,3)

%     disp([tsmall t_medium t_big t_sub])

% %     dpc

%     ok = lb>0;

