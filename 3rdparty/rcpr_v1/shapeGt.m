function varargout = shapeGt( action, varargin )
%
% Wrapper with utils for handling shape as list of landmarks
%
% shapeGt contains a number of utility functions, accessed using:
%  outputs = shapeGt( 'action', inputs );
%
% USAGE
%  varargout = shapeGt( action, varargin );
%
% INPUTS
%  action     - string specifying action
%  varargin   - depends on action
%
% OUTPUTS
%  varargout  - depends on action
%
% FUNCTION LIST
%
%%%% Model creation and visualization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   shapeGt>createModel, shapeGt>draw
%
%%%% Shape composition, inverse, distances, projection %%%%%%%%%%%%%%%
%
%   shapeGt>compose,shapeGt>inverse, shapeGt>dif, shapeGt>dist
%   shapeGt>compPhiStar, shapeGt>reprojectPose, shapeGt>projectPose
%
%%%% Shape-indexed features computation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   shapeGt>ftrsGenIm,shapeGt>ftrsCompIm
%   shapeGt>ftrsGenDup,shapeGt>ftrsComDup
%   shapeGt>ftrsOcclMasks, shapeGt>codifyPos
%   shapeGt>getLinePoint, shapeGt>getSzIm
%
%%%%% Random shape generation for initialization  %%%%%%%%%%%%%%%%%%%%
%
%   shapeGt>initTr, shapeGt>initTest
%
% EXAMPLES
%
%%create COFW model
%   model = shapeGt( 'createModel', 'cofw' );
%%draw shape on top of image
%   shapeGt( 'draw',model,Image,shape);
%%compute distance between two set of shapes (phis1 and phis2)
%   d = shapeGt( 'dist',model,phis1,phis2);
%
% For full function example usage, see individual function help and how
%  they are used in:  demoRCPR, FULL_demoRCPR, rcprTrain, rcprTest
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

varargout = cell(1,max(1,nargout));
[varargout{:}] = feval(action,varargin{:});
end

function model = createModel( type )
% Create shape model (model is necessary for all other actions).
model=struct('nfids',0,'D',0,'isFace',1,'name',[]);
switch type
    case 'cofw' % COFW dataset (29 landmarks: X,Y,V)
        model.nfids=29;model.D=model.nfids*3; model.name='cofw';
    case 'lfpw' % LFPW dataset (29 landmarks: X,Y)
        model.nfids=29;model.D=model.nfids*2; model.name='lfpw';
    case 'helen' % HELEN dataset (194 landmarks: X,Y)
        model.nfids=194;model.D=model.nfids*2;model.name='helen';
    case 'lfw' % LFW dataset (10 landmarks: X,Y)
        model.nfids=10;model.D=model.nfids*2; model.name='lfw';
    case 'pie' %Multi-pie & 300-Faces in the wild dataset (68 landmarks)
        model.nfids=68;model.D=model.nfids*2;model.name='pie';
    case 'apf' %anonimous portrait faces
        model.nfids=55;model.D=model.nfids*2;model.name='apf';
    otherwise
        error('unknown type: %s',type);
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = draw( model, Is, phis, varargin )
% Draw shape with parameters phis using model on top of image Is.
dfs={'n',25, 'clrs','gcbm', 'drawIs',1, 'lw',10, 'is',[]};
[n,cs,drawIs,lw,is]=getPrmDflt(varargin,dfs,1);

% display I
if(drawIs), im(Is); colorbar off; axis off; title(''); axis('ij'); end%clf
% special display for face model (draw face points)
hold on,
if( isfield(model,'isFace') && model.isFace ),
    [N,D]=size(phis);
    if(strcmp(model.name,'cofw')),
        %WITH OCCLUSION
        nfids = D/3;
        for n=1:N
            occl=phis(n,(nfids*2)+1:nfids*3);
            vis=find(occl==0);novis=find(occl==1);
            plot(phis(n,vis),phis(n,vis+nfids),'g.',...
                'MarkerSize',lw);
            h=plot(phis(n,novis),phis(n,novis+nfids),'r.',...
                'MarkerSize',lw);
        end
    else
        %REGULAR
        if(N==1),cs='g';end, nfids = D/2;
        for n=1:N
            h=plot(phis(n,1:nfids),phis(n,nfids+1:nfids*2),[cs(n) '.'],...
                'MarkerSize',lw);
        end
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function pos=ftrsOcclMasks(xs)
%Generate 9 occlusion masks for varied feature locations
pos=cell(9,1);
for m=1:9
    switch(m)
        case 1,pos{m}=(1:numel(xs(:,1)))';
        case 2,%top half
            pos{m}=find(xs(:,2)<=0);
        case 3,%bottom half
            pos{m}=find(xs(:,2)>0);
        case 4,%right
            pos{m}=find(xs(:,1)>=0);
        case 5,%left
            pos{m}=find(xs(:,1)<0);
        case 6,%right top diagonal
            pos{m}=find(xs(:,1)>=xs(:,2));
        case 7,%left bottom diagonal
            pos{m}=find(xs(:,1)<xs(:,2));
        case 8,%left top diagonal
            pos{m}=find(xs(:,1)*-1>=xs(:,2));
        case 9,%right bottom diagonal
            pos{m}=find(xs(:,1)*-1<xs(:,2));
    end
end
end

function ftrData = ftrsGenDup( model, varargin )
% Generate random shape indexed features, relative to
% two landmarks (points in a line, RCPR contribution)
% Features are then computed using frtsCompDup
%
% USAGE
%  ftrData = ftrsGenDup( model, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompDup

dfs={'type',4,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
F2=max(100,ceil(F*1.5));
xs=[];nfids=model.nfids;
while(size(xs,1)<F),
    %select two random landmarks
    xs(:,1:2)=randint2(F2,2,[1 nfids]);
    %make sure they are not the same
    neq = (xs(:,1)~=xs(:,2));
    xs=xs(neq,:);
end
xs=xs(1:F,:);
%select position in line
xs(:,3)=(2*radius*rand(F,1))-radius;
if(nChn>1),
    if(type==4),%make sure subbtractions occur inside same channel
        chns = randint2(F/2,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function ftrData = ftrsGenIm( model, pStar, varargin )
% Generate random shape indexed features,
% relative to closest landmark (similar to Cao et al., CVPR12)
% Features are then computed using frtsCompIm
%
% USAGE
%  ftrData = ftrsGenIm( model, pStar, varargin )
%
% INPUTS
%  model    - shape model (see createModel())
%  pStar    - average shape (see initTr)
%  varargin - additional params (struct or name/value pairs)
%   .type     - [2] feature type (1 or 2)
%   .F        - [100] number of ftrs to generate
%   .radius   - [2] sample initial x from circle of given radius
%   .nChn     - [1] number of image channels (e.g. 3 for color images)
%   .pids     - [] part ids for each x
%
% OUTPUTS
%  ftrData  - struct containing generated ftrs
%   .type     - feature type (1 or 2)
%   .F        - total number of features
%   .nChn     - number of image channels
%   .xs       - feature locations relative to unit circle
%   .pids     - part ids for each x
%
% EXAMPLE
%
% See also shapeGt>ftrsCompIm

dfs={'type',2,'F',20,'radius',1,'nChn',3,'pids',[],'mask',[]};
[type,F,radius,nChn,pids,mask]=getPrmDflt(varargin,dfs,1);
%Generate random features on image
xs1=[];
while(size(xs1,1)<F),
    xs1=rand(F*1.5,2)*2-1;
    xs1=xs1(sum(xs1.^2,2)<=1,:);
end
xs1=xs1(1:F,:)*radius;

if(strcmp(model.name,'cofw'))
    nfids=size(pStar,2)/3;
else
    nfids=size(pStar,2)/2;
end
%Reproject each into closest pStar landmark
xs=zeros(F,3);%X,Y,landmark
for f=1:F
    posX=xs1(f,1)-pStar(1:nfids);
    posY=xs1(f,2)-pStar(nfids+1:nfids*2);
    dist = (posX.^2)+(posY.^2);
    [~,l]=min(dist);xs(f,:)=[posX(l) posY(l) l];
end
if(nChn>1),
    if(mod(type,2)==0),%make sure subbtractions occur inside same channel
        chns = randint2(F,1,[1 nChn]);
        xs(1:2:end,4) = chns; xs(2:2:end,4) = chns;
    else xs(:,4)=randint2(F,1,[1 nChn]);
    end
end
if(isempty(pids)), pids=floor(linspace(0,F,2)); end
ftrData=struct('type',type,'F',F,'nChn',nChn,'xs',xs,'pids',pids);
end

function [ftrs,occlD] = ftrsCompDup( model, phis, Is, ftrData,...
    imgIds, pStar, bboxes, occlPrm)
% Compute features from ftrsGenDup on Is
%
% USAGE
%  [ftrs,Vs] = ftrsCompDup( model, phis, Is, ftrData, imgIds, pStar, ...
%       bboxes, occlPrm )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi
%  pStar   -  [1xR] average shape (see initTr)
%  bboxes   - [Nx4] face bounding boxes
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - struct containing occlusion info (if using full RCPR)
%       .group    - [MxF] to which face region each features belong
%       .featOccl - [MxF] amount of total occlusion in that area
%
% EXAMPLE
%
% See also demoRCPR, shapeGt>ftrsGenDup
N = length(Is); nfids=model.nfids;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
if(nargin<6 || isempty(pStar)),
    pStar=compPhiStar(model,phis,Is,0,[],[]);
end
M=size(phis,1); assert(length(imgIds)==M);nChn=ftrData.nChn;

if(size(bboxes,1)==length(Is)), bboxes=bboxes(imgIds,:); end

if(ftrData.type==3),
    FTot=ftrData.F;
    ftrs = zeros(M,FTot);
else
    FTot=ftrData.F;ftrs = zeros(M,FTot);
end
posrs = phis(:,nfids+1:nfids*2);poscs = phis(:,1:nfids);
useOccl=occlPrm.Stot>1;
if(useOccl && (strcmp(model.name,'cofw')))
    occl = phis(:,(nfids*2)+1:nfids*3);
    occlD=struct('featOccl',zeros(M,FTot),'group',zeros(M,FTot));
else occl = zeros(M,nfids);occlD=[];
end
%GET ALL POINTS
if(nargout>1)
    [csStar,rsStar]=getLinePoint(ftrData.xs,pStar(1:nfids),...
        pStar(nfids+1:nfids*2));
    pos=ftrsOcclMasks([csStar' rsStar']);
end
%relative to two points
[cs1,rs1]=getLinePoint(ftrData.xs,poscs,posrs);
nGroups=occlPrm.nrows*occlPrm.ncols;
%ticId =ticStatus('Computing feats',1,1,1);
for n=1:M
    img = Is{imgIds(n)}; [h,w,ch]=size(img);
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    cs1(n,:)=max(1,min(w,cs1(n,:)));
    rs1(n,:)=max(1,min(h,rs1(n,:)));
    
    %where are the features relative to bbox?
    if(useOccl && (strcmp(model.name,'cofw')))
        %to which group (zone) does each feature belong?
        occlD.group(n,:)=codifyPos((cs1(n,:)-bboxes(n,1))./bboxes(n,3),...
            (rs1(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %to which group (zone) does each landmark belong?
        groupF=codifyPos((poscs(n,:)-bboxes(n,1))./bboxes(n,3),...
            (posrs(n,:)-bboxes(n,2))./bboxes(n,4),...
            occlPrm.nrows,occlPrm.ncols);
        %NEW
        %therefore, what is the occlusion in each group (zone)
        occlAm=zeros(1,nGroups);
        for g=1:nGroups
            occlAm(g)=sum(occl(n,groupF==g));
        end
        %feature occlusion = sum of occlusion on that area
        occlD.featOccl(n,:)=occlAm(occlD.group(n,:));
    end
    
    inds1 = (rs1(n,:)) + (cs1(n,:)-1)*h;
    if(nChn>1), inds1 = inds1+(chs'-1)*w*h; end
    
    if(isa(img,'uint8')), ftrs1=double(img(inds1)')/255;
    else ftrs1=double(img(inds1)'); end
    
    if(ftrData.type==3),ftrs1=ftrs1*2-1; ftrs(n,:)=reshape(ftrs1,1,FTot);
    else ftrs(n,:)=ftrs1;
    end
    %tocStatus(ticId,n/M);
end
end

function group=codifyPos(x,y,nrows,ncols)
%codify position of features into regions
nr=1/nrows;nc=1/ncols;
%Readjust positions so that they falls in [0,1]
x=min(1,max(0,x));y=min(1,max(0,y));
y2=y;x2=x;
for c=1:ncols,
    if(c==1), x2(x<=nc)=1;
    elseif(c==ncols), x2(x>=nc*(c-1))=ncols;
    else x2(x>nc*(c-1) & x<=nc*c)=c;
    end
end
for r=1:nrows,
    if(r==1), y2(y<=nr)=1;
    elseif(r==nrows), y2(y>=nc*(r-1))=nrows;
    else y2(y>nr*(r-1) & y<=nr*r)=r;
    end
end
group=sub2ind2([nrows ncols],[y2' x2']);
end

function [cs1,rs1]=getLinePoint(FDxs,poscs,posrs)
%get pixel positions given coordinates as points in a line between
%landmarks
%INPUT NxF, OUTPUT NxF
if(size(poscs,1)==1)%pStar normalized
    l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
    x1 = poscs(:,l1);y1 = posrs(:,l1);
    x2 = poscs(:,l2);y2 = posrs(:,l2);
    
    a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
    distX=(x2-x1)/2; ctrX= x1+distX;
    cs1=ctrX+(repmat(xs',size(distX,1),1).*distX);
    rs1=(a.*cs1)+b;
else
    if(size(FDxs,2)<4)%POINTS IN A LINE (ftrsGenDup)
        %2 points in a line with respect to center
        l1= FDxs(:,1);l2= FDxs(:,2);xs=FDxs(:,3);
        %center
        muX = mean(poscs,2);
        muY = mean(posrs,2);
        poscs=poscs-repmat(muX,1,size(poscs,2));
        posrs=posrs-repmat(muY,1,size(poscs,2));
        %landmark
        x1 = poscs(:,l1);y1 = posrs(:,l1);
        x2 = poscs(:,l2);y2 = posrs(:,l2);
        
        a=(y2-y1)./(x2-x1); b=y1-(a.*x1);
        distX=(x2-x1)/2; ctrX= x1+distX;
        cs1=ctrX+(repmat(xs',size(distX,1),1).*distX);
        rs1=(a.*cs1)+b;
        cs1=round(cs1+repmat(muX,1,size(FDxs,1)));
        rs1=round(rs1+repmat(muY,1,size(FDxs,1)));
    end
end
end

function [ftrs,occlD] = ftrsCompIm( model, phis, Is, ftrData,...
    imgIds, pStar, bboxes, occlPrm )
% Compute features from ftrsGenIm on Is
%
% USAGE
%  [ftrs,Vs] = ftrsCompIm( model, phis, Is, ftrData, [imgIds] )
%
% INPUTS
%  model    - shape model
%  phis     - [MxR] relative shape for each image
%  Is       - cell [N] input images [w x h x nChn] variable w, h
%  ftrData  - define ftrs to actually compute, output of ftrsGen
%  imgIds   - [Mx1] image id for each phi
%  pStar   -  [1xR] average shape (see initTr)
%  bboxes   - [Nx4] face bounding boxes
%  occlPrm   - struct containing occlusion reasoning parameters
%    .nrows - [3] number of rows in face region
%    .ncols - [3] number of columns in face region
%    .nzones - [1] number of zones from where each regs draws
%    .Stot -  total number of regressors at each stage
%    .th - [0.5] occlusion threshold used during cascade
%
% OUTPUTS
%  ftrs     - [MxF] computed features
%  occlD    - [] empty structure
%
% EXAMPLE
%
% See also demoCPR, shapeGt>ftrsGenIm, shapeGt>ftrsCompDup

N = length(Is); nChn=ftrData.nChn;
if(nargin<5 || isempty(imgIds)), imgIds=1:N; end
M=size(phis,1); assert(length(imgIds)==M);

[pStar,phisN,distPup,sz,bboxes] = ...
    compPhiStar( model, phis, Is, 10, imgIds, bboxes );

if(size(bboxes,1)==length(Is)), bboxes=bboxes(imgIds,:); end

F=size(ftrData.xs,1);ftrs = zeros(M,F);
useOccl=occlPrm.Stot>1;
if(strcmp(model.name,'cofw'))
    nfids=size(phis,2)/3;occlD=[];
else
    nfids=size(phis,2)/2;occlD=[];
end

%X,Y,landmark,Channel
rs = ftrData.xs(:,2);cs = ftrData.xs(:,1);xss = [cs';rs'];
ls = ftrData.xs(:,3);if(nChn>1),chs = ftrData.xs(:,4);end
%Actual phis positions
poscs=phis(:,1:nfids);posrs=phis(:,nfids+1:nfids*2);
%get positions of key landmarks
posrs=posrs(:,ls);poscs=poscs(:,ls);
%Reference points
X=[pStar(1:nfids);pStar(nfids+1:nfids*2)];
for n=1:M
    img = Is{imgIds(n)}; [h,w,ch]=size(img);
    if(ch==1 && nChn==3), img = cat(3,img,img,img);
    elseif(ch==3 && nChn==1), img = rgb2gray(img);
    end
    
    %Compute relation between phisN and pStar (scale, rotation)
    Y=[phisN(n,1:nfids);phisN(n,nfids+1:nfids*2)];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [~,~,Sc,Rot] = translate_scale_rotate(Y,X);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Compute feature locations by reprojecting
    aux=Sc*Rot*xss;
    %Resize accordingly to bbox size
    szX=bboxes(n,3)/2;szY=bboxes(n,4)/2;
    aux = [(aux(1,:)*szX);(aux(2,:)*szY)];
    
    %Add to respective landmark
    rs1 = round(posrs(n,:)+aux(2,:));
    cs1 = round(poscs(n,:)+aux(1,:));
    
    cs1 = max(1,min(w,cs1)); rs1=max(1,min(h,rs1));
    
    inds1 = (rs1) + (cs1-1)*h;
    if(nChn>1), chs = repmat(chs,1,m); inds1 = inds1+(chs-1)*w*h; end
    
    if(isa(img,'uint8')), ftrs1=double(img(inds1)')/255;
    else ftrs1=double(img(inds1)'); end
    
    if(ftrData.type==1),
        ftrs1=ftrs1*2-1; ftrs(n,:)=reshape(ftrs1,1,F);
    else ftrs(n,:)=ftrs1;
    end
end
end

function [h,w]=getSzIm(Is)
%get image sizes
N=length(Is); w=zeros(1,N);h=zeros(1,N);
for i=1:N, [w(i),h(i),~]=size(Is{i}); end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function phis = compose( model, phis0, phis1, bboxes )
% Compose two shapes phis0 and phis1: phis = phis0 + phis1.
phis1=projectPose(model,phis1,bboxes);
phis=phis0+phis1;
end

function phis = inverse( model, phis0, bboxes )
% Compute inverse of two shapes phis0 so that phis0+phis=phis+phis0=identity.
phis=-projectPose(model,phis0,bboxes);
end

function [phiStar,phisN,distPup,sz,bboxes] = ...
    compPhiStar( model, phis, Is, pad, imgIds, bboxes )
% Compute phi that minimizes sum of distances to phis (average shape)
[N,D] = size(phis);sz=zeros(N,2);
if(isempty(imgIds)),imgIds=1:N; end

if(strcmp(model.name,'cofw')), nfids = (D/3);
else nfids=D/2;
end

phisN=zeros(N,D);
if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
    distPup=sqrt(((phis(:,17)-phis(:,18)).^2)+...
        ((phis(:,17+nfids)-phis(:,18+nfids)).^2));
elseif(strcmp(model.name,'mouseP'))
    distPup=68;
elseif(strcmp(model.name,'lfw'))
    leyeX=mean(phis(:,1:2),2);leyeY=mean(phis(:,(1:2)+nfids),2);
    reyeX=mean(phis(:,7:8),2);reyeY=mean(phis(:,(7:8)+nfids),2);
    distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
else distPup=0;
end
if(nargin<6), bboxes = zeros(N,4); end
for n=1:N
    if(nargin<6)
        %left top width height
        bboxes(n,1:2)=[min(phis(n,1:nfids))-pad ...
            min(phis(n,nfids+1:end))-pad];
        bboxes(n,3)=max(phis(n,1:nfids))-bboxes(n,1)+2*pad;
        bboxes(n,4)=max(phis(n,nfids+1:nfids*2))-bboxes(n,2)+2*pad;
    end
    img = Is{imgIds(n)}; [sz(n,1),sz(n,2),~]=size(img);
    sz(n,:)=sz(n,:)/2;
    %All relative to centroid, using bbox size
    if(nargin<6)
        szX=bboxes(n,3)/2;szY=bboxes(n,4)/2;
        ctr(1)=bboxes(n,1)+szX;ctr(2) = bboxes(n,2)+szY;
    else
        szX=bboxes(imgIds(n),3)/2;szY=bboxes(imgIds(n),4)/2;
        ctr(1)=bboxes(imgIds(n),1)+szX;ctr(2) = bboxes(imgIds(n),2)+szY;
    end
    
    if(strcmp(model.name,'cofw'))
        phisN(n,:) = [(phis(n,1:nfids)-ctr(1))./szX ...
            (phis(n,nfids+1:nfids*2)-ctr(2))./szY ...
            phis(n,(nfids*2)+1:nfids*3)];
    else phisN(n,:) = [(phis(n,1:nfids)-ctr(1))./szX ...
            (phis(n,nfids+1:nfids*2)-ctr(2))./szY];
    end
end
phiStar = mean(phisN,1);
end

function phis1=reprojectPose(model,phis,bboxes)
%reproject shape given bounding box of object location
[N,D]=size(phis);
if(strcmp(model.name,'cofw')), nfids = D/3;
else nfids = D/2;
end
szX=bboxes(:,3)/2;szY=bboxes(:,4)/2;
ctrX = bboxes(:,1)+szX;ctrY = bboxes(:,2)+szY;
szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);
ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);
if(strcmp(model.name,'cofw'))
    phis1 = [(phis(:,1:nfids).*szX)+ctrX (phis(:,nfids+1:nfids*2).*szY)+ctrY...
        phis(:,(nfids*2)+1:nfids*3)];
else
    phis1 = [(phis(:,1:nfids).*szX)+ctrX (phis(:,nfids+1:nfids*2).*szY)+ctrY];
end
end

function phis=projectPose(model,phis,bboxes)
%project shape onto bounding box of object location
[N,D]=size(phis);
if(strcmp(model.name,'cofw')), nfids=D/3;
else nfids=D/2;
end
szX=bboxes(:,3)/2;szY=bboxes(:,4)/2;
ctrX=bboxes(:,1)+szX;ctrY=bboxes(:,2)+szY;
szX=repmat(szX,1,nfids);szY=repmat(szY,1,nfids);
ctrX=repmat(ctrX,1,nfids);ctrY=repmat(ctrY,1,nfids);
if(strcmp(model.name,'cofw'))
    phis = [(phis(:,1:nfids)-ctrX)./szX (phis(:,nfids+1:nfids*2)-ctrY)./szY ...
        phis(:,(nfids*2)+1:nfids*3)];
else  phis = [(phis(:,1:nfids)-ctrX)./szX (phis(:,nfids+1:nfids*2)-ctrY)./szY];
end
end

function del = dif( phis0, phis1 )
% Compute diffs between phis0(i,:,t) and phis1(i,:) for each i and t.
[N,R,T]=size(phis0); assert(size(phis1,3)==1);
del = phis0-phis1(:,:,ones(1,1,T));
end

function [ds,dsAll] = dist( model, phis0, phis1 )
% Compute distance between phis0(i,:,t) and phis1(i,:) for each i and t.
%relative to the distance between pupils in the image (phis1 = gt)
[N,R,T]=size(phis0); del=dif(phis0,phis1);
if(strcmp(model.name,'cofw'))
    nfids = size(phis1,2)/3;
else nfids = size(phis1,2)/2;
end

%Distance between pupils
if(strcmp(model.name,'lfpw') || strcmp(model.name,'cofw'))
    distPup=sqrt(((phis1(:,17)-phis1(:,18)).^2) + ...
        ((phis1(:,17+nfids)-phis1(:,18+nfids)).^2));
    distPup = repmat(distPup,[1,nfids,T]);
elseif(strcmp(model.name,'lfw'))
    leyeX=mean(phis1(:,1:2),2);leyeY=mean(phis1(:,(1:2)+nfids),2);
    reyeX=mean(phis1(:,7:8),2);reyeY=mean(phis1(:,(7:8)+nfids),2);
    distPup=sqrt(((leyeX-reyeX).^2) + ((leyeY-reyeY).^2));
    distPup = repmat(distPup,[1,nfids,T]);
elseif(strcmp(model.name,'helen'))
    leye = [mean(phis1(:,135:154),2) mean(phis1(:,nfids+(135:154)),2)];
    reye = [mean(phis1(:,115:134),2) mean(phis1(:,nfids+(115:134)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,nfids,T]);
elseif(strcmp(model.name,'pie'))
    leye = [mean(phis1(:,37:42),2) mean(phis1(:,nfids+(37:42)),2)];
    reye = [mean(phis1(:,43:48),2) mean(phis1(:,nfids+(43:48)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,nfids,T]);
elseif(strcmp(model.name,'apf'))
    leye = [mean(phis1(:,7:8),2) mean(phis1(:,nfids+(7:8)),2)];
    reye = [mean(phis1(:,9:10),2) mean(phis1(:,nfids+(9:10)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
elseif(strcmp(model.name,'aflw_frontal'))
    leye = [mean(phis1(:,12),2) mean(phis1(:,nfids+(12)),2)];
    reye = [mean(phis1(:,23),2) mean(phis1(:,nfids+(23)),2)];
    distPup=sqrt(((reye(:,1)-leye(:,1)).^2)+...
        ((reye(:,2)-leye(:,2)).^2));
    distPup = repmat(distPup,[1,nfids,T]);
elseif(strcmp(model.name,'aflw_profile'))
    distPup = 40;
    distPup = repmat(distPup,[size(phis1,1),nfids,T]);
end
dsAll = sqrt((del(:,1:nfids,:).^2) + (del(:,nfids+1:nfids*2,:).^2));
dsAll = dsAll./distPup; ds=mean(dsAll,2);%2*sum(dsAll,2)/R;
end

function [pCur,pGt,pGtN,pStar,imgIds,N,N1]=initTr(Is,pGt,...
    model,pStar,posInit,L,pad)
%Randomly initialize each training image with L shapes
[N,D]=size(pGt);assert(length(Is)==N);
if(isempty(pStar)),
    [pStar,pGtN]=compPhiStar(model,pGt,Is,pad,[],posInit);
end
% augment data amount by random permutations of initial shape
pCur=repmat(pGt,[1,1,L]);
if(strcmp(model.name,'cofw'))
    nfids = size(pGt,2)/3;
else nfids = size(pGt,2)/2;
end
for n=1:N
    %select other images
    imgsIds = randSample([1:n-1 n+1:N],L);%[n randSample(1:N,L-1)];
    %Project onto image
    for l=1:L
        %permute bbox location slightly
        maxDisp = posInit(n,3:4)/16;
        uncert=(2*rand(1,2)-1).*maxDisp;
        bbox=posInit(n,:);bbox(1:2)=bbox(1:2)+uncert;
        pCur(n,:,l)=reprojectPose(model,pGtN(imgsIds(l),:),bbox);
    end
end
if(strcmp(model.name,'cofw'))
    pCur = reshape(permute(pCur,[1 3 2]),N*L,nfids*3);
else pCur = reshape(permute(pCur,[1 3 2]),N*L,nfids*2);
end

imgIds=repmat(1:N,[1 L]);pGt=repmat(pGt,[L 1]);pGtN=repmat(pGtN,[L 1]);
N1=N; N=N*L;
end

function p=initTest(Is,bboxes,model,pStar,pGtN,RT1,my_inds)
%Randomly initialize testing shapes using training shapes (RT1 different)
N=length(Is);D=size(pStar,2);phisN=pGtN;

if(isempty(bboxes)), p=pStar(ones(N,1),:);
    %One bbox provided per image
elseif(ismatrix(bboxes) && size(bboxes,2)==4),
    p=zeros(N,D,RT1);NTr=size(phisN,1);%gt=regModel.phisT;
    for n=1:N
        %select other images
        if (nargin == 7 && ~isempty(my_inds))
            imgsIds = my_inds;
        else
            imgsIds = randSample(NTr,RT1);
        end
        %Project into image
        for l=1:RT1
            %permute bbox location slightly (not scale)
            maxDisp = bboxes(n,3:4)/16;            
            uncert=(2*rand(1,2)-1).*maxDisp;
            bbox=bboxes(n,:);bbox(1:2)=bbox(1:2)+uncert;
            p(n,:,l)=reprojectPose(model,phisN(imgsIds(l),:),bbox);
        end
    end
    %RT1 bboxes given, just reproject
elseif(size(bboxes,2)==4 && size(bboxes,3)==RT1)
    p=zeros(N,D,RT1);NTr=size(phisN,1);
    for n=1:N
        imgsIds = randSample(NTr,RT1);
        for l=1:RT1
            p(n,:,l)=reprojectPose(model,phisN(imgsIds(l),:),...
                bboxes(n,:,l));
        end
    end
    %Previous results are given, use as is
elseif(size(bboxes,2)==D && size(bboxes,3)==RT1)
    p=bboxes;
    %VIDEO
elseif(iscell(bboxes))
    p=zeros(N,D,RT1);NTr=size(pGtN,1);
    for n=1:N
        bb=bboxes{n}; ndet=size(bb,1);
        imgsIds = randSample(NTr,RT1);
        if(ndet<RT1), bbsIds=randint2(1,RT1,[1,ndet]);
        else bbsIds=1:RT1;
        end
        for l=1:RT1
            p(n,:,l)=reprojectPose(model,pGtN(imgsIds(l),:),...
                bb(bbsIds(l),1:4));
        end
    end
end
end
