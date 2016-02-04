function p = rcprTest1( Is, regModel, p, regPrm, iniData, ...
    verbose, prunePrm)
% Apply robust cascaded shape regressor.
%
% USAGE
%  p = rcprTest1( Is, regModel, p, regPrm, bboxes, verbose, prunePrm)
%
% INPUTS
%  Is       - cell(N,1) input images
%  regModel - learned multi stage shape regressor (see rcprTrain)
%  p        - [NxDxRT1] initial shapes
%  regPrm   - struct with regression parameters (see regTrain)
%  iniData  - [Nx2] or [Nx4] bbounding boxes/initial positions
%  verbose  - [1] show progress or not 
%  prunePrm - [REQ] parameters for smart restarts 
%     .prune     - [0] whether to use or not smart restarts
%     .maxIter   - [2] number of iterations
%     .th        - [.15] threshold used for pruning 
%     .tIni      - [10] iteration from which to prune
%
% OUTPUTS
%  p        - [NxD] shape returned by multi stage regressor
%
% EXAMPLE
%
% See also rcprTest, rcprTrain
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

% Apply each single stage regressor starting from shape p.
model=regModel.model; T=regModel.T; [N,D,RT1]=size(p);
p=reshape(permute(p,[1 3 2]),[N*RT1,D]);
imgIds = repmat(1:N,[1 RT1]); regs = regModel.regs;

%Get prune parameters
maxIter=prunePrm.maxIter;prune=prunePrm.prune;
th=prunePrm.th;tI=prunePrm.tIni;

%Set up data
p_t=zeros(size(p,1),D,T+1);p_t(:,:,1)=p;
if(model.isFace),bbs=iniData(imgIds,:,1);else bbs=[];end
done=0;Ntot=0;k=0;
N1=N;p1=p;imgIds1=imgIds;pos=1:N;md=zeros(N,RT1,D,maxIter);
%Iterate while not finished
while(~done)
    %Apply cascade
    tStart=clock;
    %If pruning is active, each loop returns the shapes of the examples
    %that passed the smart restart threshold (good) and 
    %those that did not (bad)
    [good1,bad1,p_t1,p1,p2]=cascadeLoop(Is,model,regModel,regPrm,T,N1,D,RT1,...
        p1,imgIds1,regs,tStart,iniData,bbs,verbose,...
        prune,1,th,tI);
    %Separate into good/bad (smart restarts)
    good=pos(good1);mem=ismember(imgIds,good);
    p_t(mem,:,:)=p_t1;p(mem,:)=p1; 
    Ntot=Ntot+length(good); done=Ntot==N; 
    if(~done)
        %Keep iterating only on bad
        Is=Is(bad1);N1=length(bad1);pos=pos(bad1);
        imgIds1 = repmat(1:N1,[1 RT1]);
        if(model.isFace) 
            iniData=iniData(bad1,:,:);bbs=iniData(imgIds1,:,1); 
            p1=shapeGt('initTest',Is,iniData,...
                model,regModel.pStar,regModel.pGtN,RT1);
            p1=reshape(permute(p1,[1 3 2]),[N1*RT1,D]);
        else
            iniData=iniData(bad1,:,:);
            p1=shapeGt('initTest',Is,model,...
                iniData,regModel.pStar,RT1);
            p1=reshape(permute(p1,[1 3 2]),[N1*RT1,D]);
        end
        
        md(pos,:,:,k+1)=reshape(p2,[N1,RT1,D]);
        k=k+1;
        %If maxIter has been reached, use median of all 
        if(k>=maxIter)
            RT=RT1*maxIter;
            p1=reshape(permute(md(pos,:,:,:),[1 3 2 4]),[N1,D,RT]);
            %select initialization for each indepently based on distance
            dist=zeros(N1,RT);dist2=zeros(N1,RT);
            for r=1:RT
                aux = permute(shapeGt('dist',model,p1(:,:,:),p1(:,:,r)),[1 3 2]);
                close=zeros(N1,RT);
                close(aux<th)=1;
                dist(:,r)=sum(close,2); 
                try dist2(:,r)=mean(aux,2);
                catch 
                    warning('not enough shapes');size(aux),size(dist2)
                end
            end
            %expand to RT1 different initializations
            p2=zeros(N1,D,RT1);
            for n=1:N1
                ind=find(dist(n,:)>=RT*0.7);%4
                if(length(ind)>=RT1)
                    use=randSample(ind,RT1);
                    p2(n,:,:)=p1(n,:,use);
                else
                    [~,ix]=sort(dist2(n,:));
                    p2(n,:,:)=p1(n,:,ix(1:RT1));
                end
            end
            %Call cascade loop one last time 
            p1=reshape(permute(p2,[1 3 2]),[N1*RT1,D]);tStart=clock;
            [~,~,p_t1,p1,~]=cascadeLoop(Is,model,regModel,regPrm,T,N1,D,RT1,...
                p1,imgIds1,regs,tStart,iniData,bbs,verbose,...
                0,tI,th,tI);
            remain=pos; ind=ismember(imgIds,remain);
            p_t(ind,:,:)=p_t1;p(ind,:)=p1;
            done=1;
        end
    end
end
%reconvert p from [N*RT1xD] to [NxDxRT1]
p=permute(reshape(p,[N,RT1,D]),[1 3 2]);
%p_t=permute(reshape(p_t,[N,RT1,D,T+1]),[1 3 2 4]);
end

%Apply full RCPR cascade with check in between if smart restart is enabled
function [good,bad,p_t,p,p2]=cascadeLoop(Is,model,regModel,regPrm,T,N,D,RT1,p,...
    imgIds,regs,tStart,bboxes,bbs,verbose,prune,t0,th,tI)

p_t=zeros(size(p,1),D,T+1);p_t(:,:,1)=p;
good=1:N;bad=[];p2=[];
for t=t0:T
    %Compute shape-indexed features
    ftrPos=regs(t).ftrPos;
    if(ftrPos.type>2)
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompDup',model,p,Is,ftrPos,...
            imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
    else
        [ftrs,regPrm.occlD] = shapeGt('ftrsCompIm',model,p,Is,ftrPos,...
            imgIds,regModel.pStar,bboxes,regPrm.occlPrm);
    end
    %Retrieve learnt regressors 
    regt=regs(t).regInfo;
    %Apply regressors
    p1=shapeGt('projectPose',model,p,bbs);
    pDel=regApply(p1,ftrs,regt,regPrm);
    p=shapeGt('compose',model,pDel,p,bbs);
    p=shapeGt('reprojectPose',model,p,bbs);
    p_t(:,:,t+1)=p;
    %If reached checkpoint, check state of restarts   
    if((prune && T>tI && t==tI))
       [p_t,p,good,bad,p2]=checkState(p_t,model,imgIds,N,t,th,RT1);
       if(isempty(good)),return; end
       Is=Is(good);N=length(good);imgIds=repmat(1:N,[1 RT1]);
       if(model.isFace),bboxes=bboxes(good,:);bbs=bboxes(imgIds,:);end
    end
    if((t==1 || mod(t,5)==0) && verbose)
        msg=tStatus(tStart,t,T);fprintf(['Applying ' msg]); 
    end
end
end

function [p_t,p,good,bad,p2]=checkState(p_t,model,imgIds,N,t,th,RT1)
    %Confidence computation=variance between different restarts
    %If output has low variance and low distance, continue (good)
    %ow recurse with new initialization (bad)
    p=permute(p_t(:,:,t+1),[3 2 1]);conf=zeros(N,RT1);
    for n=1:N
        pn=p(:,:,imgIds==n);md=median(pn,3);
        %variance=distance from median of all predictions
        conf(n,:)=shapeGt('dist',model,pn,md);
    end
    dist=mean(conf,2);
    bad=find(dist>th);good=find(dist<=th);
    p2=p_t(ismember(imgIds,bad),:,t+1);
    p_t=p_t(ismember(imgIds,good),:,:);p=p_t(:,:,t+1);
    if(isempty(good)),return; end
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
msg = ['  [elapsed=' elptimeS timetypeS ...
    ' / remaining~=' esttimeS timetypeS ']\n' ];
end