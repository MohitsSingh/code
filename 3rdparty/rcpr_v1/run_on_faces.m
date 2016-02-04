% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion,
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For pre-requisites and compilation, see CONTENTS.m
%
% This code tests a pre-trained RCPR (data/rcpr.mat) on COFW dataset.
%  COFW test is composed of one file (data/COFW_test.mat)
%  which contains:
%    -phisT - ground truth shapes
%    -IsT - images
%    -bboxesT - face bounding boxes
%  If you change path to folder containing training/testing files, change
%  this variable here:
COFW_DIR='./data/';
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
addpath('~/code/utils');
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COFW dataset (test only)
% testing images and ground truth
trainFile=[COFW_DIR 'COFW_train.mat'];
load(trainFile,'phisTr','IsTr','bboxesTr');bboxesT=round(bboxesT);
nfids=size(phisT,2)/3;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD PRE-TRAINED RCPR model
load([COFW_DIR 'rcpr.mat'],'regModel','regPrm','prunePrm')
% load ~/storage/misc/face_images_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST
%Initialize randomly using RT1 shapes drawn from training
RT1=10;
phisT_bu = phisT;
% my code
n = 0;
ff = 1.2;
xys = {};
            
[res,bb] = detect_on_set(face_images_1,regModel,bboxesTr,regPrm,[]);
save ~/storage/misc/face_images_xy res bb
% for k = 1:length(face_images_1) % 143
%     k
%
%     if (length(xys) >= k && ~isempty(xys{k}))
%         clf; imagesc2(face_images_1{k});
%         plot(xys{k}(:,1),xys{k}(:,2),'g+');
%         pause;continue;
%     end
%
%     bb = [];
%     IsT = {};
%     resizeRatio = 80/size(face_images_1{k},1);
%     IsT{1} = imResample(im2uint8(face_images_1{k}),resizeRatio,'bilinear');
%     d = .2;
%     sz = size(IsT{1});
%     if isempty(bb)
%         bboxesT = [d*sz(2) d*sz(1) (1-2*d)*sz(2) (1-2*d)*sz(1)];
%     else
%         bboxesT = bb;
%     end
%     p=shapeGt('initTest',IsT(1),bboxesT,regModel.model,...
%         regModel.pStar,regModel.pGtN,RT1);
%     testPrm = struct('RT1',RT1,'pInit',bboxesT,...
%         'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
%         'verbose',1);
%     t=clock;[p,pRT] = rcprTest(IsT(1),regModel,testPrm);t=etime(clock,t);
%     %Round up the pixel positions
%     p(:,1:nfids*2)=round(p(:,1:nfids*2));
%     % Use threshold computed during training to
%     % binarize occlusion
%     occl=p(:,(nfids*2)+1:end);
%     occl(occl>=regModel.th)=1;occl(occl<regModel.th)=0;
%     p(:,(nfids*2)+1:end)=occl;
%     nimage=1;
%     xy = [p(1:29);p(30:58)]';
%     xys{k} = xy/resizeRatio;
% end
% save ~/storage/misc/face_images_1_xy res bb