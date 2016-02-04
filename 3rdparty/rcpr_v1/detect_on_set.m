
function [res,bb] = detect_on_set(face_images,regModel,regPrm,prunePrm,bbs)

%% TEST
%Initialize randomly using RT1 shapes drawn from training
RT1=30;
% phisT_bu = phisT;
n = 0;
ff = 1.2;
res = {};

for k = 1:length(face_images)
    k
%     bb = [];
   % resizeRatio = 200/size(face_images{k},1);
    resizeRatio = 100/size(face_images{k},1);
    
    I = imResample(im2uint8(face_images{k}),resizeRatio,'bilinear');
    d = .2;
    sz = size(I);
    if isempty(bbs)
        bboxesT = [d*sz(2) d*sz(1) (1-d)*sz(2) (1-d)*sz(1)];
    else
        bb = bbs(k,:)*resizeRatio;
        bboxesT = bb;
    end
    p=shapeGt('initTest',{I},bboxesT,regModel.model,...
        regModel.pStar,regModel.pGtN,RT1);
    testPrm = struct('RT1',RT1,'pInit',bboxesT,...
        'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
        'verbose',1);
    t=clock;[p,pRT] = rcprTest({I},regModel,testPrm);t=etime(clock,t);
    nfids = 29;
    p(:,1:nfids*2)=round(p(:,1:nfids*2));
    % Use threshold computed during training to
    % binarize occlusion
    occl=p(:,(nfids*2)+1:end);
    occl(occl>=regModel.th)=1;occl(occl<regModel.th)=0;
    p(:,(nfids*2)+1:end)=occl;
    clf;shapeGt('draw',regModel.model,I,p); drawnow;pause(.01);
    plotBoxes(bboxesT);
    p(1:58) = p(1:58)/resizeRatio;
    res{k} = p;
    bb = bboxesT; bb(3:4) = bb(3:4)+bb(1:2); bb = bb/resizeRatio;
end