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
COFW_DIR='./data/';
addpath(genpath('/home/amirro/code/3rdparty/piotr_toolbox/'));
addpath('~/code/utils');
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COFW dataset (test only)
% testing images and ground truth
testFile=[COFW_DIR 'COFW_test.mat'];
load(testFile,'phisT','IsT','bboxesT');bboxesT=round(bboxesT);
nfids=size(phisT,2)/3;
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD PRE-TRAINED RCPR model
% load([COFW_DIR 'rcpr.mat'],'regModel','regPrm','prunePrm')
% load ~/storage/misc/face_images_1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TEST
%Initialize randomly using RT1 shapes drawn from training
RT1=30;
phisT_bu = phisT;
% my code
n = 0;
ff = 1;
for k = 1:length(face_images_1) % 143            
    bb = [];
    for u = 1
    
    IsT = {};
    
   %% 
    face_images_1 = {I};
    IsT{1} = (imResample(im2uint8(face_images_1{k}),ff^(u-1)*100/size(face_images_1{k},1),'bilinear'));
    d = .2;
    sz = size(IsT{1});
    if isempty(bb)                
        bboxesT = [d*sz(2) d*sz(1) (1-d)*sz(2) (1-d)*sz(1)];
%         bboxesT = [(d+.01)*sz(2) d*sz(1) (1-2*d)*sz(2) (1-2*d)*sz(1)];
    else
        bboxesT = bb;
    end
    bboxesT_v = bboxesT;
    bboxesT_v(3:4) = bboxesT(3:4)-bboxesT(1:2);
%     bboxesT_v(1) = bboxesT_v(1) -5;
    
    
    %bboxesT = bboxesT(1,:);
    % end of my code
    %
    RT1 =10;
    p=shapeGt('initTest',IsT(1),bboxesT_v,regModel.model,...
        regModel.pStar,regModel.pGtN,RT1);
    
    testPrm = struct('RT1',RT1,'pInit',bboxesT_v,...
        'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
        'verbose',1);
    
    %Test
%     testPrm = rmfield(testPrm,'initData');
    t=clock;[p,pRT] = rcprTest(IsT(1),regModel,testPrm);t=etime(clock,t);
    %Round up the pixel positions
    %p(:,1:nfids*2)=round(p(:,1:nfids*2));
    % Use threshold computed during training to
    % binarize occlusion
    occl=p(:,(nfids*2)+1:end);
    occl(occl>=regModel.th)=1;occl(occl<regModel.th)=0;
    p(:,(nfids*2)+1:end)=occl;
    
    %
    %Compute loss
    % loss = shapeGt('dist',regModel.model,p,phisT);
    % fprintf('--------------DONE\n');
    %
    % VISUALIZE Example results on a test image
    figure(1),clf,
    nimage=1;
    %Ground-truth
    % subplot(1,2,1),
    % shapeGt('draw',regModel.model,IsT{nimage},phisT(nimage,:),...
    %     {'lw',20});
    % title('Ground Truth');
    %Prediction
    xy = [p(1:29);p(30:58)]';
    shapeGt('draw',regModel.model,IsT{nimage},p(nimage,:),...
        {'lw',20});
    plotBoxes(bboxesT)
%     bboxesT_v = bboxesT;bboxesT_v(3:4) = bboxesT(3:4)+bboxesT(1:2);
%     plotBoxes(bboxesT_v);
%     plot(xy(:,1),xy(:,2),'r+');
    %%
    pause;continue
    
    bb = ff*pts2Box(xy);
    
    plotBoxes(pts2Box(xy),'m--','LineWidth',2);
    bb(3:4) = bb(3:4)-bb(1:2);
    title('Prediction');
    u
    pause
    end
end