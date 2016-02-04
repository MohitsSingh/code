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

opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_classes_full_pascal/net-epoch-50';
%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-fcnTrain_action_objects_8_single_obj_full/net-epoch-94.mat';
%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-fcnTrain_action_region_full/net-epoch-28.mat';
% opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
% opts.modelPath  = '/net/mraid11/export/data/amirro/fcn/data/fcn8s_person_parts_focused/net-epoch-33.mat';
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat
load ~/storage/misc/pascal_action_imdb.mat
imdb = pascal_imdb;
imdb.images = 1:length(imdb.images_data);
train = find(imdb.isTrain);
val = find(~imdb.isTrain);

%     labels = {};
% classLabels = {'none','face','hand',...
%     'phoning',...
%     'playinginstrument',...
%     'reading',...
%     'ridingbike',...
%     'ridinghorse',...
%     'running',...
%     'takingphoto',...
%     'usingcomputer',...
%     'walking'};
imdb.nClasses = 7; % 
opts.useGpu = 1;
% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------
% gpuDevice(1)
[net_full_image,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath, opts.modelFamily);
net_full_image.move('gpu');

% sub-net
zoom_path = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_classes_full_pascal_zoom/net-epoch-60'
[net_sub_patch,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(zoom_path, opts.modelFamily);
net_sub_patch.move('gpu');


%%
load ~/storage/misc/pascal_action_imdb.mat

%%
% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
% imdb.nClasses = 3;
% numGpus = 0 ;
% nClasses = imdb.nClasses+1;
% confusion = zeros(nClasses) ;
%%
preds = {};
scores = {};
addpath('~/code/3rdparty/subplot_tight');
%%
outPath = '~/storage/pascal_action_fcn_results/';
ensuredir(outPath);
labels_full{8} = 'nnn';
for i = 1:1:numel(imdb.images)
    i
%     if imdb.isTrain(i),continue,end
%     if none(imdb.labels{i}>0),continue,end
    %     if max(col(imdb.labels{i}))<=12,continue,end
    max(col(imdb.labels{i}))
    imgName = pascal_imdb.imageIds{i};
    curBoxes = pascal_imdb.img_boxes{i};
    imId = i;
    curResPath = fullfile(outPath,[imgName '.mat']);
    if exist(curResPath,'file'),continue,end
    %if ~any(col(imdb.labels{imId})==10),continue,end
    rgb = single(imdb.images_data{imId});
    if size(rgb,3)==1
        rgb = cat(3,rgb,rgb,rgb);
    end
    rgb_orig = rgb;
    sz_orig = size2(rgb);
    rgb = imResample(rgb,[384 384],'bilinear');
    %     predict_and_show(net_full_image,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,true,labels_full);
    %     dpc
    %     continue
    [pred_full,scores_full_image] = predict_and_show(net_full_image,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,false,labels_full);
    tsmall = toc;
    % get subpatches and show prediction for each sub-patch
    scores_full_image = imResample(scores_full_image,sz_orig,'bilinear');
    pred_full = imResample(pred_full,sz_orig,'nearest');
    softmaxScores_full = bsxfun(@rdivide,exp(scores_full_image),sum(exp(scores_full_image),3));
    
    fra_sub_patches = false;
    if fra_sub_patches
        
        %         softmaxScores_full = bsxfun(@rdivide,exp(scores_full_image),sum(exp(scores_full_image),3));
        M = max(softmaxScores_full(:,:,2:end),[],3);
        % go over sub-windows and aggerate results.
        [subs,vals] = nonMaxSupr( double(M), 20, 0,10 );
        subs = fliplr(subs);
        boxes = round(inflatebbox(subs,mean(sz_orig)/3,'both',true));
        
    else
        bb = curBoxes;
        bb = makeSquare(bb);
        boxes = round(inflatebbox(bb,1.2,'both',false));
    end
    curPatches = multiCrop2(rgb_orig,boxes);
    % %     x2(cellfun2(@(x) x/255,curPatches));
    [scores_hires,t_sub] = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch);
    %     [scores_hires_lm,t_sub] = doHighResScan(rgb_orig,boxes,curPatches,net_sub_patch);
    [~,preds_full_res] = max(scores_hires,[],3);
    
    % % %     showPredictions(rgb_orig,pred_full,softmaxScores_full,labels_full,1);
    % % %     showPredictions(rgb_orig,preds_full_res,scores_hires,labels_full,2);
    % % % % %
    % % % % %     box_scores = {};
    % % % % %     for iBox = 1:size(curBoxes,1)
    % % % % %         bb = curBoxes(iBox,:);
    % % % % %         bb = makeSquare(bb);
    % % % % %         bb = round(inflatebbox(bb,1.2,'both',false));
    % % % % %         curSubImg = cropper(rgb_orig,bb);
    % % % % %         curSubImg = imResample(curSubImg,[384 384],'bilinear');
    % % % % %         [pred_sub,scores_sub] = predict_and_show(net_sub_patch,imageNeedsToBeMultiple,inputVar,'prediction',single(curSubImg),2,true,labels_full);
    % % % % %         box_scores{iBox} = scores_sub;
    % %         dpc
    % % % % %     end
    % % % % %
    % % % % %
    % dpc
%     continue
    %         dpc;
    
    %     scores_hires_sm = bsxfun(@rdivide,exp(scores_hires),sum(exp(scores_hires),3));
    %     showPredictions(rgb_orig,preds_full_res,scores_hires,labels_full,2)
    %     dpc    
        
    save(curResPath,'scores_full_image','boxes','scores_hires');
    
%     save(curResPath,'scores_full_image','curBoxes','imgName','box_scores');
end

