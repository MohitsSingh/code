% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup

addpath utils
% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;
% opts.modelType = 'fcn32s' ;
% opts.expDir = 'data/fcn32s-voc11' ;
% opts.dataDir = 'data/voc11' ;
%opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn32s-action_obj/net-epoch-50.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro//fcn/data/fcn8s-action_obj_8/net-epoch-100.mat';
% opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s-action_obj_8_x2_w_hand s/net-epoch-100.mat';
%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s_person_parts/net-epoch-50.mat';
%%
opts.modelPath = '/net/mraid11/export/data/amirro/fcn/data/fcn8s_person_parts_focused_face_and_hand/net-epoch-115.mat';
tvgPath = '/net/mraid11/export/data/amirro/matconv_data/pascal-fcn8s-tvg-dag.mat';
net_tvg = dagnn.DagNN.loadobj(load(tvgPath)) ;
net_tvg.mode = 'test' ;
predVar_tvg = 'coarse' ;
inputVar_tvg = 'data' ;
imageNeedsToBeMultiple = false ;
net_tvg.move('gpu');

%%
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% load ~/code/mircs/images_and_face_obj.mat;
load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat
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
imdb.nClasses = 24;

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

switch opts.modelFamily
    case 'matconvnet'
        net = load(opts.modelPath) ;
        net = dagnn.DagNN.loadobj(net.net) ;
        net.mode = 'test' ;
        for name = {'objective', 'accuracy'}
            net.removeLayer(name) ;
        end
        net.meta.normalization.averageImage = reshape(net.meta.normalization.rgbMean,1,1,3) ;
        predVar = net.getVarIndex('prediction') ;
        inputVar = 'input' ;
        imageNeedsToBeMultiple = true ;
    case 'ModelZoo'
        net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
        net.mode = 'test' ;
        predVar = net.getVarIndex('upscore') ;
        inputVar = 'data' ;
        imageNeedsToBeMultiple = false ;
    case 'TVG'
        net = dagnn.DagNN.loadobj(load(opts.modelPath)) ;
        net.mode = 'test' ;
        predVar = net.getVarIndex('coarse') ;
        inputVar = 'data' ;
        imageNeedsToBeMultiple = false ;
end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------
imdb.nClasses = 24;
numGpus = 0 ;
nClasses = imdb.nClasses+1;
confusion = zeros(nClasses) ;
net.move('gpu');
% pimap = part2ind();
% M = pimap{15};
%
% all_keys = M.keys;
% all_inds = [];
% labels = {};
% for t = 1:length(all_keys)
%    labels{M(all_keys{t})} = all_keys{t};
% end
%
% labels = {'none',labels{:}};
% labels.keys()
%%
% LUT = zeros(size(LUT))
%     LUT(ismember(LUT_OLD,[1 2 3 4 5 6 7])) = 1 % face
%     LUT(ismember(LUT_OLD,12)) = 2 % hand
%     LUT(ismember(LUT_OLD,10:11))= 3;% hands/arms
%     LUT(ismember(LUT_OLD,13:15))= 4;% legs
%     LUT(ismember(LUT_OLD,8))= 5;% torso
%
L =load('~/storage/misc/voc10_action_trainval.mat');
% test = randperm(length(L.action_gt_train));
labels= {'none','face','hand','arms','legs','torso'};
% p = randperm(length(fra_db));
%test = randperm(length(L.action_gt_train));
test = 1:length(L.action_gt_train);
imgDir = '/home/amirro/storage/data/Stanford40//JPEGImages/';

load ~/code/mircs/fra_db_2015_10_08.mat

imgDir=  '/home/amirro/storage/data/willowactions/JPEGImages/';
d = getAllFiles(imgDir,'*.jpg');

%%
for i = 1:1:numel(d)
    i
    imId = d{i};
    %     if isempty(strfind(imId,'drink')),continue,end
    rgb = vl_imreadjpeg({imId}); rgb = rgb{1};
    %
    % %     [rgb,curBox] = getImage(conf,imId);
    % %     rgb_orig = single(255*rgb);
    % %     curBox = inflatebbox(curBox,1.2,'both',false);
    %     curBox = round(curBox);
    %     plotBoxes(curBox);
    %     rgb = cropper(rgb_orig,curBox);
    %     if size(rgb,1)<384
    %
    %         rgb=  imResample(rgb,384/size(rgb,1),'bilinear');
    %     end
    %     %
    %     predict_and_show(net,true,inputVar,'prediction',rgb,2,true,labels);
    %     dpc; continue;
    % rgb = vl_imreadjpeg({j2m(imgDir,imId,'.jpg')});rgb = rgb{1};
    %rgb = vl_imreadjpeg({j2m(imgDir,fra_db(i),'.jpg')});rgb = rgb{1};
    
    %
    %     for u = 1:10
    %         u
    %         rgb = rgb+50*abs(rand(size(rgb)));
    %         rgb = rgb-min(rgb(:));
    %         rgb = 255*rgb/max(rgb(:));
    % % % % %         pred_tvg = predict_and_show(net_tvg,imageNeedsToBeMultiple,inputVar_tvg,predVar_tvg,rgb,2,true,net_tvg.meta.classes.name);
    %         dpc
    %     end
    %x2(pred_tvg==16)
    %     curBox = curImgData.personBox;
    %     curBox = makeSquare(curImgData.personBox,true);
    %     curBox = round(inflatebbox(curBox,1,'both',false));
    %     curBox = round(curBox);
    %     rgb = cropper(rgb,curBox);
    %rgb = imResample(rgb,[384/size 384],'bilinear');
    % for i = 1:30:1215
    %     imId = i;%test(i) ;
    %     y = getBatch_action_obj(imdb, imId,'labelOffset',1,'useFlipping',false,'doScaling',false);
    %     rgb = y{2};
    %     anno = y{4};
    %     lb = single(anno);
    
    
    rgb_orig = rgb;
    
    L = bwlabel(pred_tvg==16);
    for t = 1:max(L(:))
        
        curBox = region2Box(L==t);
        
        [a,b,c] = BoxSize(curBox);
        if c < 2000
            continue
        end
        
        
        
        figure(1); clf; imagesc2(rgb_orig/255);
        %curBox = round(makeSquare(curBox,true));
        curBox = inflatebbox(curBox,1.2,'both',false);
        curBox = round(curBox);
        plotBoxes(curBox);
        rgb = cropper(rgb_orig,curBox);
        if size(rgb,1)<384
            
            rgb=  imResample(rgb,384/size(rgb,1),'bilinear');
        end
        %
        predict_and_show(net,true,inputVar,'prediction',rgb,2,true,labels);
        dpc;
    end
end

% for i = 1:1:numel(test)
%     i
%     imId = test(i) ;
%     curImgData = L.action_gt_train(imId);
%     if ~curImgData.actions.reading,continue,end
%         imagePath = fullfile(L.vocImgDir,[curImgData.imageID '.jpg']);
%      rgb = vl_imreadjpeg({imagePath});rgb = rgb{1};
% %     rgb = vl_imreadjpeg({j2m(imgDir,fra_db(i),'.jpg')});rgb = rgb{1};
%
%     curBox = curImgData.personBox;
%     curBox = makeSquare(curImgData.personBox,true);
%     curBox = round(inflatebbox(curBox,1,'both',false));
%     curBox = round(curBox);
%     rgb = cropper(rgb,curBox);
%      rgb = imResample(rgb,[384 384],'bilinear');
% % for i = 1:30:1215
% %     imId = i;%test(i) ;
% %     y = getBatch_action_obj(imdb, imId,'labelOffset',1,'useFlipping',false,'doScaling',false);
% %     rgb = y{2};
% %     anno = y{4};
% %     lb = single(anno);
%     predict_and_show(net,imageNeedsToBeMultiple,inputVar,'prediction',rgb,1,true,labels);
%     dpc;
% end
