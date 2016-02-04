function [regModel,pAll] = rcprTrain( Is, pGt, varargin )
% Train multistage robust cascaded shape regressor
%
% USAGE
%  [regModel,pAll] = rcprTrain( Is, pGt, varargin )
%
% INPUTS
%  Is       - cell(N,1) input images
%  pGt      - [NxR] ground truth shape for each image
%  varargin - additional params (struct or name/value pairs)
%   .model    - [REQ] shape model
%   .pStar    - [] initial shape
%   .posInit  - [] known object position (e.g. tracking output)
%   .T        - [REQ] number of stages
%   .L        - [1] data augmentation factor
%   .regPrm   - [REQ] param struct for regTrain
%   .ftrPrm   - [REQ] param struct for shapeGt>ftrsGen
%   .regModel - [Tx1] previously learned single stage shape regressors
%   .pad      - amount of padding around bbox 
%   .verbose  - [0] method verbosity during training
%   .initData - initialization parameters (see shapeGt>initTr)
%
% OUTPUTS
%  regModel - learned multi stage shape regressor:
%   .model    - shape model
%   .pStar    - [1xR] average shape 
%   .pDstr    - [NxR] ground truth shapes
%   .T        - number of stages
%   .pGtN     - [NxR] normalized ground truth shapes
%   .th       - threshold for occlusion detection
%   .regs     - [Tx1] struct containing learnt cascade of regressors
%      .regInfo  - [KxStot] regressors
%         .ysFern  - [2^MxR] fern bin averages
%         .thrs    - [Mx1] thresholds
%         .fids    - [2xM] features used
%      .ftrPos   - feature information
%         .type    - type of features
%         .F       - number of features
%         .nChn    - number of channels used
%         .xs      - [Fx3] features position
%         .pids    - obsolete
%
%  pAll     - shape estimation at each iteration T
%
% EXAMPLE
%
% See also  demoRCPR, FULL_demoRCPR
%
% Copyright 2013 X.P. Burgos-Artizzu, P.Perona and Piotr Dollar.  
%  [xpburgos-at-gmail-dot-com]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the Simplified BSD License [see bsd.txt]
%
%  Please cite our paper if you use the code:
%  Robust face landmark estimation under occlusion, 
%  X.P. Burgos-Artizzu, P. Perona, P. Dollar (c)
%  ICCV'13, Sydney, Australia

% get additional parameters and check dimensions
dfs={'model','REQ','pStar',[],'posInit',[],'T','REQ',...
    'L',1,'regPrm','REQ','ftrPrm','REQ','regModel',[],...
    'pad',10,'verbose',0,'initData',[]};
[model,pStar,posInit,T,L,regPrm,ftrPrm,regModel,pad,verbose,initD] = ...
    getPrmDflt(varargin,dfs,1);

[regModel,pAll]=rcprTrain1(Is, pGt,model,pStar,posInit,...
        T,L,regPrm,ftrPrm,regModel,pad,verbose,initD);
end 

function [regModel,pAll]=rcprTrain1(Is, pGt,model,pStar,posInit,...
    T,L,regPrm,ftrPrm,regModel,pad,verbose,initD)
% Initialize shape and assert correct image/ground truth format
if(isempty(initD))
    [pCur,pGt,pGtN,pStar,imgIds,N,N1]=shapeGt('initTr',Is,pGt,...
        model,pStar,posInit,L,pad);
else
    pCur=initD.pCur;pGt=initD.pGt;pGtN=initD.pGtN;
    pStar=initD.pStar;imgIds=initD.imgIds;N=initD.N;N1=initD.N1;
    clear initD;
end
D=size(pGt,2);
% remaining initialization, possibly continue training from
% previous model
pAll = zeros(N1,D,T+1);
regs = repmat(struct('regInfo',[],'ftrPos',[]),T,1);
if(isempty(regModel)), t0=1; pAll(:,:,1)=pCur(1:N1,:);
else
    t0=regModel.T+1; regs(1:regModel.T)=regModel.regs;
    [~,pAll1]=cprApply(Is,regModel,'imgIds',imgIds,'pInit',pCur);
    pAll(:,:,1:t0)=pAll1(1:N1,:,:); pCur=pAll1(:,:,end);
end

loss = mean(shapeGt('dist',model,pCur,pGt));
if(verbose), 
    fprintf('  t=%i/%i       loss=%f     ',t0-1,T,loss); 
end
tStart = clock;%pCur_t=zeros(N,D,T+1);
bboxes=posInit(imgIds,:);
for t=t0:T
    % get target value for shape
    pTar = shapeGt('inverse',model,pCur,bboxes);
    pTar = shapeGt('compose',model,pTar,pGt,bboxes);
    
    if(ftrPrm.type>2)
        ftrPos = shapeGt('ftrsGenDup',model,ftrPrm);
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup',...
            model,pCur,Is,ftrPos,...
            imgIds,pStar,posInit,regPrm.occlPrm);
    else
        ftrPos = shapeGt('ftrsGenIm',model,pStar,ftrPrm);
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompIm',...
            model,pCur,Is,ftrPos,...
            imgIds,pStar,posInit,regPrm.occlPrm);
    end
    
    %Regress
    regPrm.ftrPrm=ftrPrm;
    [regInfo,pDel] = regTrain(ftrs,pTar,regPrm);
    pCur = shapeGt('compose',model,pDel,pCur,bboxes);
    pCur = shapeGt('reprojectPose',model,pCur,bboxes);
    
    pAll(:,:,t+1)=pCur(1:N1,:);
    %loss scores
    loss = mean(shapeGt('dist',model,pCur,pGt));
    % store result
    regs(t).regInfo=regInfo;
    regs(t).ftrPos=ftrPos;
    %If stickmen, add part info
    if(verbose),
        msg=tStatus(tStart,t,T);
        fprintf(['  t=%i/%i       loss=%f     ' msg],t,T,loss); 
    end
    if(loss<1e-5), T=t; break; end
end
% create output structure
regs=regs(1:T); pAll=pAll(:,:,1:T+1);
regModel = struct('model',model,'pStar',pStar,...
    'pDstr',pGt(1:N1,:),'T',T,'regs',regs);
if(~strcmp(model.name,'ellipse')),regModel.pGtN=pGtN(1:N1,:); end
% Compute precision recall curve for occlusion detection and find 
% desired occlusion detection performance (default=90% precision)
if(strcmp(model.name,'cofw'))
    nfids=D/3;
    occlGt=pGt(:,(nfids*2)+1:end);
    op=pCur(:,(nfids*2)+1:end);
    indO=find(occlGt==1);
    
    th=0:.01:1;
    prec=zeros(length(th),1);
    recall=zeros(length(th),1);
    for i=1:length(th)
        indPO=find(op>th(i));
        prec(i)=length(find(occlGt(indPO)==1))/numel(indPO);
        recall(i)=length(find(op(indO)>th(i)))/numel(indO);
    end
    %precision around 90% (or closest)
    pos=find(prec>=0.9);
    if(~isempty(pos)),pos=pos(1);
    else [~,pos]=max(prec);
    end
    %maximum f1score
    % f1score=(2*prec.*recall)./(prec+recall);    
    % [~,pos]=max(f1score);
    regModel.th=th(pos);
end
end

function msg=tStatus(tStart,t,T)
elptime = etime(clock,tStart);
fracDone = max( t/T, .00001 );
esttime = elptime/fracDone - elptime;
if( elptime/fracDone < 600 )
    elptimeS  = num2str(elptime,'%.1f');
    esttimeS  = num2str(esttime,'%.1f');
    timetypeS = 's';
else
    elptimeS  = num2str(elptime/60,'%.1f');
    esttimeS  = num2str(esttime/60,'%.1f');
    timetypeS = 'm';
end
msg = ['[elapsed=' elptimeS timetypeS ...
    ' / remaining~=' esttimeS timetypeS ']\n' ];
end