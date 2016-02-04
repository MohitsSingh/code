
%%
frontal.zhu_nose_center = 6;
frontal.zhu_left_mouth = 35;
frontal.zhu_right_mouth = 44;
frontal.zhu_chin_center = 52;
frontal.zhu_left_eye = 12;
frontal.zhu_right_eye = 22;
frontal.zhu = [6 35 44 53 12 22];
side.zhu_nose_center = 3;
side.zhu_left_mouth = 18;
side.zhu_right_mouth = 18;
side.zhu_chin_center = 30;
side.zhu_left_eye = 10;
side.zhu = [3 18 18 30 10];

%%
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
% This code trains and tests RCPR on COFW dataset.
%  COFW is composed of two files (data/COFW_train.mat, data/COFW_test.mat)
%  which contain:
%    -phisTr,phisT - ground truth shapes (train/test)
%    -IsTr,IsT - images (train/test)
%    -bboxesTr, bboxesT - face bounding boxes (train/test)
%  If you change path to folder containing training/testing files, change
%  this variable here:
COFW_DIR='./data/';
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOAD COFW dataset
% training/testing images and ground truth
trFile=[COFW_DIR 'COFW_train.mat'];
testFile=[COFW_DIR 'COFW_test.mat'];
% Load files
load(trFile,'phisTr','IsTr','bboxesTr');bboxesTr=round(bboxesTr);
load(testFile,'phisT','IsT','bboxesT');bboxesT=round(bboxesT);
nfids=size(phisTr,2)/3;

addpath ~/code/mircs
initpath;
config
addpath(genpath('~/code/utils'))

%%%% amir 26/1/2015 - create training dataset from e.g, aflw.
x2(%T_score = 2.45; % minimal face detection score...

T_score = 1; % minimal face detection score...
im_subset = row(find(scores > T_score));
%         im_subset = vl_colsubset(im_subset,10000,'random');
curImgs = ims(im_subset);
requiredKeypoints = {pts.pointNames};
requiredKeypoints = cat(1,requiredKeypoints{:});
requiredKeypoints = unique(requiredKeypoints);
all_kps = getKPCoordinates_2(pts(im_subset),requiredKeypoints)+1;
isnan(all_kps,3)
%nnz(any(any(isnan(all_kps),3),2))
for t = 1:size(all_kps,1)
    clf; imagesc2(curImgs{t});
    curPts = squeeze(all_kps(t,:,:));    
    nnz(any(isnan(curPts),2))
    plotPolygons(1+squeeze(all_kps(t,:,:)),'g+');
    drawnow
    pause
end

%%%%


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET UP PARAMETERS
%Choose algorithm to use
% cpr_type=1 (reimplementation of Cao et al.)
% cpr_type=2 RCPR (features+restarts)
% cpr_type=3 RCPR (full)
cpr_type=3;
if(cpr_type==1)
    %Remove occlusion information
    phisTr=phisTr(:,1:nfids*2);phisT=phisT(:,1:nfids*2);
    %Create LFPW model (29 landmarks without visibility)
    model = shapeGt('createModel','lfpw');
    %CPR for face PARAMETERS (Cao et al. CVPR12)
    %(type 2, points relative to closest landmark)
    T=100;K=50;L=20;RT1=5;
    ftrPrm = struct('type',2,'F',400,'nChn',1,'radius',1);
    prm=struct('thrr',[-1 1]/5,'reg',.01);
    occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);
    regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
        'loss','L2','R',0,'M',5,'model',model,'prm',prm);
    prunePrm=struct('prune',0,'maxIter',2,'th',0.15,'tIni',10);
elseif(cpr_type==2)
    %Remove occlusion information
    phisTr=phisTr(:,1:nfids*2);phisT=phisT(:,1:nfids*2);
    %Create LFPW model (29 landmarks without visibility)
    model = shapeGt('createModel','lfpw');
    %RCPR(features+restarts) PARAMETERS
    %(type 4, points relative to any 2 landmarks)
    T=100;K=50;L=20;RT1=5;
    ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
    prm=struct('thrr',[-1 1]/5,'reg',.01);
    occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);
    regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
        'loss','L2','R',0,'M',5,'model',model,'prm',prm);
    %smart restarts are enabled
    prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);
    %remove occlusion information
    phisTr=phisTr(:,1:nfids*2);phisT=phisT(:,1:nfids*2);
    %Create LFPW model (29 landmarks without visibility)
    model = shapeGt('createModel','lfpw');
elseif(cpr_type==3)
    %Create COFW model (29 landmarks including visibility)
    %model = shapeGt('createModel','cofw');
    model = shapeGt('createModel','helen');
    %RCPR (full) PARAMETERS
    %(type 4, points relative to any 2 landmarks, radius 2)
    T=100;K=15;L=20;RT1=5;
    ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
    prm=struct('thrr',[-1 1]/5,'reg',.01);
    %Stot=3 regressors to perform occlusion weighted median
    occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',3,'th',.5);
    regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
        'loss','L2','R',0,'M',5,'model',model,'prm',prm);
    %smart restarts are enabled
    prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);    
elseif(cpr_type==4) % my data, no visibility
    
    inDir = '~/storage/aflw_zhu';
    load ~/storage/misc/all_lm_aflw_zhu.mat % all_lm, all_inds, d
    %%
    % frontal model
    min_score =0.4;
    nReqKeypoints = 68; % for profile, require 39
    scores = [all_lm.s];
    nKP = arrayfun(@(x) size(x.xy,1),all_lm);    
    comp_range = [4:9];
    %comps = [all_lm.c];    
    cur_lm = all_lm(scores >= min_score & nKP==nReqKeypoints);
    cur_inds = all_inds(nKP==nReqKeypoints & scores >= min_score);
    ss_1 = [cur_lm.s];               
    [r,ir] = sort(ss_1,'descend');
    %%
    faces_and_landmarks_frontal = struct('I',{},'xy',{},'c',{});
    n = 0;
    %for u = length(ir):-1:1 % subsample...                        
    for u = 1:1:length(ir) % subsample...                        
        n = n+1;
        %length(ir):-50:1
        k = ir(u);
        
        %if k > 1000,continue,end                
        
%         if cur_lm(k).rotation ==0,continue,end
        num2str([u r(u)])
        curInd = str2num(d(cur_inds(k)).name(1:end-4));        
        I = ims{curInd};
%         U = imrotate(I,cur_lm(k).rotation,'bilinear','crop');
        faces_and_landmarks_frontal(n).I = I;
        faces_and_landmarks_frontal(n).bbox = round(inflatebbox([1 1 fliplr(size2(I))],1/1.3,'both',false));
        xy = boxCenters(cur_lm(k).xy);
        xy = rotate_pts(xy,-pi*cur_lm(k).rotation/180,size2(I)/2);
        faces_and_landmarks_frontal(n).xy = xy;
        faces_and_landmarks_frontal(n).c = cur_lm(k).c;
        faces_and_landmarks_frontal(n).s = cur_lm(k).s;                       
% % %         I = faces_and_landmarks_frontal(n).I;
% % %         clf; imagesc2(I);
% % %         plotPolygons(faces_and_landmarks_frontal(n).xy,'r+');
% % %         plotBoxes(faces_and_landmarks_frontal(n).bbox);
% % %         drawnow
% % %         pause(.01)
% %                     pause
    end    
    
    
    %%
%     comps = unique([faces_and_landmarks_frontal.c]);
    
    %%
        
    faces_and_landmarks_train = faces_and_landmarks_frontal(1:1:end);
    %mImage({faces_and_landmarks_frontal.I});
    model = shapeGt('createModel','lfpw');
    model.isFace = 1;
    model.name = 'aflw_frontal';
    model.nfids = 68;
    model.D = 136;
    nTrain = length(faces_and_landmarks_train);
    phisTr = zeros(nTrain,model.D);
    for u = 1:length(faces_and_landmarks_train)
        phisTr(u,:) = reshape(faces_and_landmarks_train(u).xy,[],1);
    end
    IsTr = {faces_and_landmarks_train.I};
    bboxesTr = cat(1,faces_and_landmarks_train.bbox);
    bboxesTr(:,3:4) = bboxesTr(:,3:4)-bboxesTr(:,1:2);
% %     
% %     %
% %     for t = 1:length(IsTr)
% %         clf; imagesc2(IsTr{t});
% %         x = phisTr(t,1:end/2);
% %         y = phisTr(t,end/2+1:end);
% %         plot(x,y,'g+');
% %         plot(x(12),y(12),'m*');
% %         plot(x(23),y(23),'c*');
% %         pause;
% %     end
% %     
    %shapeGt('draw',model,faces_and_landmarks_train(1).I,phisTr(1,:),{'lw',15});
    %CPR for face PARAMETERS (Cao et al. CVPR12)
    %(type 2, points relative to closest landmark)
    T=100;K=50;L=20;RT1=5;
    ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
    prm=struct('thrr',[-1 1]/5,'reg',.01);
    occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);
    regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
        'loss','L2','R',0,'M',5,'model',model,'prm',prm);
    %smart restarts are enabled
    prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN
%Initialize randomly L shapes per training image
[pCur,pGt,pGtN,pStar,imgIds,N,N1]=shapeGt('initTr',...
    IsTr,phisTr,model,[],bboxesTr,L,10);
initData=struct('pCur',pCur,'pGt',pGt,'pGtN',pGtN,'pStar',pStar,...
    'imgIds',imgIds,'N',N,'N1',N1);
%Create training structure
trPrm=struct('model',model,'pStar',[],'posInit',bboxesTr,...
    'T',T,'L',L,'regPrm',regPrm,'ftrPrm',ftrPrm,...
    'pad',10,'verbose',1,'initData',initData);
%Train model
[regModel_frontal,~] = rcprTrain(IsTr,phisTr,trPrm);
save('regModel_frontal_2015.mat','regModel_frontal');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
faces_and_landmarks_test = faces_and_landmarks(2:5:end);
nTest = length(faces_and_landmarks_test);
phisT = zeros(nTest,D);
for u = 1:length(faces_and_landmarks_test)
    phisT(u,:) = reshape(faces_and_landmarks_test(u).xy,[],1);
end
IsT = {faces_and_landmarks_test.I};
bboxesT = cat(1,faces_and_landmarks_test.bbox);
bboxesT(:,3:4) = bboxesT(:,3:4)-bboxesT(:,1:2);

%% TEST
%Initialize randomly using RT1 shapes drawn from training
p=shapeGt('initTest',IsT,bboxesT,model,pStar,pGtN,RT1);
%Create test struct
testPrm = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
    'verbose',1);
%Test
t=clock;[p,pRT] = rcprTest(IsT,regModel,testPrm);t=etime(clock,t);
%Round up the pixel positions
p(:,1:nfids*2)=round(p(:,1:nfids*2));
% If rcpr_type=3, use threshold computed during training to
% binarize occlusion
%Compute loss
loss = shapeGt('dist',regModel.model,p,phisT);
fprintf('--------------DONE\n');

regModel_frontal = regModel;
save regModel_frontal regModel_frontal
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DISPLAY Standard histogram of errors
figure(1),clf,
mu1=mean(loss(loss<0.1));muAll=mean(loss);
fail=100*length(find(loss>0.1))/length(loss);
bins=log10(min(loss)):0.1:log10(max(loss));ftsz=20;
[n,b]=hist(log10(loss),bins); n=n./sum(n);
semilogx(10.^b,n,'b','LineWidth',3);
hold on,plot(zeros(10,1)+2.5,linspace(0,max(n),10),'--k');
ticks=[0 linspace(min(loss),max(loss)/4,5) ...
    linspace((max(loss)/3),max(loss),3)];
ticks=round(ticks*100)/100;
set(gca,'XTick',ticks,'FontSize',ftsz);
xlabel('error','FontSize',ftsz);ylabel('probability','FontSize',ftsz),
title(['Mean error=' num2str(muAll,'%0.2f') '   ' ...
    'Mean error (<0.1)=' num2str(mu1,'%0.2f') '   ' ...
    'Failure rate (%)=' num2str(fail,'%0.2f')],'FontSize',ftsz);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZE Example results on a test image

for nimage=1:100
    figure(3),clf,
    %Ground-truth
    subplot(1,2,1),
    shapeGt('draw',model,IsT{nimage},phisT(nimage,:),{'lw',20});
    title('Ground Truth');
    %Prediction
    subplot(1,2,2),shapeGt('draw',model,IsT{nimage},p(nimage,:),...
        {'lw',20});
    title('Prediction');
    pause
end


%% now for the profile....
nReqKeypoints = 39; % for profile, require 39
scores = [all_lm.s];
nKP = arrayfun(@(x) size(x.xy,1),all_lm);
min_score = .4;
cur_lm = all_lm(scores >= min_score & nKP==nReqKeypoints);
cur_inds = all_inds(nKP==nReqKeypoints & scores >= min_score);
ss_1 = [cur_lm.s];
[r,ir] = sort(ss_1,'descend');
faces_and_landmarks_profile = struct('I',{},'xy',{},'c',{});
n = 0;
for u = length(ir):-1:1 % subsample...
    n = n+1;
    %length(ir):-50:1
    k = ir(u);
    num2str([u r(u)])
    curInd = str2num(d(cur_inds(k)).name(1:end-4));
    I = ims{curInd};
    %U = imrotate(I,cur_lm(k).rotation,'bilinear','crop');
    
    faces_and_landmarks_profile(n).I = I;
    faces_and_landmarks_profile(n).bbox = round(inflatebbox([1 1 fliplr(size2(I))],1/1.3,'both',false));
    
    xy = boxCenters(cur_lm(k).xy);
    xy = rotate_pts(xy,-pi*cur_lm(k).rotation/180,size2(I)/2);
    faces_and_landmarks_profile(n).xy = xy;
    
%     faces_and_landmarks_profile(n).xy = boxCenters(cur_lm(k).xy);
    faces_and_landmarks_profile(n).c = cur_lm(k).c;
    wasFlipped = false;
    if (cur_lm(k).c > 10)
        disp('flipping');
        wasFlipped  = true;
        faces_and_landmarks_profile(n).c = 1;
        faces_and_landmarks_profile(n).I = flip_image(I);
        %faces_and_landmarks_profile(n).bbox = flip_box(faces_and_landmarks_profile(n).bbox,I);
        faces_and_landmarks_profile(n).xy(:,1) = size(I,2)-faces_and_landmarks_profile(n).xy(:,1);
    else
        %             disp('not flipping');
    end
% %     if (wasFlipped )        
%     clf;
%     imagesc2(faces_and_landmarks_profile(n).I);
%     plotPolygons(faces_and_landmarks_profile(n).xy,'r+');
%     plotBoxes(faces_and_landmarks_profile(n).bbox);
%     drawnow;pause
% %     end
end
save ~/training_landmarks.mat faces_and_landmarks_profile



save ~/storage/misc/landmark_training_data faces_and_landmarks_frontal faces_and_landmarks_profile
%%
%     comps = [faces_and_landmarks_profile.c];
%     [c,ic] = sort(comps,'ascend');
%     mImage({faces_and_landmarks_profile(ic).I});
%     figure,plot(c)

faces_and_landmarks_train = faces_and_landmarks_profile(1:1:end);
%mImage({faces_and_landmarks.I});
model = shapeGt('createModel','lfpw');
model.isFace = 1;
model.name = 'aflw_profile';
model.nfids = 39;
model.D = model.nfids*2;
nTrain = length(faces_and_landmarks_train);
phisTr = zeros(nTrain,model.D);
for u = 1:length(faces_and_landmarks_train)
    u
    phisTr(u,:) = reshape(faces_and_landmarks_train(u).xy,[],1);
end
IsTr = {faces_and_landmarks_train.I};
bboxesTr = cat(1,faces_and_landmarks_train.bbox);
bboxesTr(:,3:4) = bboxesTr(:,3:4)-bboxesTr(:,1:2);
% %
for t = 1:length(IsTr)
    clf; imagesc2(IsTr{t});
    x = phisTr(t,1:end/2);
    y = phisTr(t,end/2+1:end);
    plot(x,y,'g+');
    plot(x(10),y(10),'m*');
    plot(x(23),y(23),'c*');
    pause;
end

%shapeGt('draw',model,faces_and_landmarks_train(1).I,phisTr(1,:),{'lw',15});
%CPR for face PARAMETERS (Cao et al. CVPR12)
%(type 2, points relative to closest landmark)
T=100;K=50;L=20;RT1=5;
ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
prm=struct('thrr',[-1 1]/5,'reg',.01);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',1,'th',.5);
regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',model,'prm',prm);
%smart restarts are enabled
prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% TRAIN
%Initialize randomly L shapes per training image
[pCur,pGt,pGtN,pStar,imgIds,N,N1]=shapeGt('initTr',...
    IsTr,phisTr,model,[],bboxesTr,L,10);
initData=struct('pCur',pCur,'pGt',pGt,'pGtN',pGtN,'pStar',pStar,...
    'imgIds',imgIds,'N',N,'N1',N1);
%Create training structure
trPrm=struct('model',model,'pStar',[],'posInit',bboxesTr,...
    'T',T,'L',L,'regPrm',regPrm,'ftrPrm',ftrPrm,...
    'pad',10,'verbose',1,'initData',initData);
%Train model
[regModel_profile,~] = rcprTrain(IsTr,phisTr,trPrm);
save regModel_profile_2015 regModel_profile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%
faces_and_landmarks_test = faces_and_landmarks(2:2:end);
nTest = length(faces_and_landmarks_test);
phisT = zeros(nTest,model.D);
for u = 1:length(faces_and_landmarks_test)
    phisT(u,:) = reshape(faces_and_landmarks_test(u).xy,[],1);
end
IsT = {faces_and_landmarks_test.I};
bboxesT = cat(1,faces_and_landmarks_test.bbox);
bboxesT(:,3:4) = bboxesT(:,3:4)-bboxesT(:,1:2);

%% TEST
%Initialize randomly using RT1 shapes drawn from training
p=shapeGt('initTest',IsT,bboxesT,model,pStar,pGtN,RT1);
%Create test struct
testPrm = struct('RT1',RT1,'pInit',bboxesT,...
    'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
    'verbose',1);
%Test
t=clock;[p,pRT] = rcprTest(IsT,regModel_profile,testPrm);t=etime(clock,t);
%Round up the pixel positions
p(:,1:model.nfids*2)=round(p(:,1:model.nfids*2));
% If rcpr_type=3, use threshold computed during training to
% binarize occlusion
%Compute loss
loss = shapeGt('dist',regModel_profile.model,p,phisT);
fprintf('--------------DONE\n');

% regModel_profile = regModel;
save regModel_profile regModel_profile
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DISPLAY Standard histogram of errors
figure(1),clf,
mu1=mean(loss(loss<0.1));muAll=mean(loss);
fail=100*length(find(loss>0.1))/length(loss);
bins=log10(min(loss)):0.1:log10(max(loss));ftsz=20;
[n,b]=hist(log10(loss),bins); n=n./sum(n);
semilogx(10.^b,n,'b','LineWidth',3);
hold on,plot(zeros(10,1)+2.5,linspace(0,max(n),10),'--k');
ticks=[0 linspace(min(loss),max(loss)/4,5) ...
    linspace((max(loss)/3),max(loss),3)];
ticks=round(ticks*100)/100;
set(gca,'XTick',ticks,'FontSize',ftsz);
xlabel('error','FontSize',ftsz);ylabel('probability','FontSize',ftsz),
title(['Mean error=' num2str(muAll,'%0.2f') '   ' ...
    'Mean error (<0.1)=' num2str(mu1,'%0.2f') '   ' ...
    'Failure rate (%)=' num2str(fail,'%0.2f')],'FontSize',ftsz);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% VISUALIZE Example results on a test image
for nimage=1:100
    figure(3),clf,
    %Ground-truth
    subplot(1,2,1),
    shapeGt('draw',model,IsT{nimage},phisT(nimage,:),{'lw',20});
    title('Ground Truth');
    %Prediction
    subplot(1,2,2),shapeGt('draw',model,IsT{nimage},p(nimage,:),...
        {'lw',20});
    title('Prediction');
    pause
end
%% apply to fra_db....
load ~/storage/mircs_18_11_2014/s40_fra;
fra_db = s40_fra;
initpath
config
t = 1;

load regModel_frontal_2015
load regModel_profile_2015
T=100;K=15;L=20;RT1=5;
ftrPrm = struct('type',4,'F',400,'nChn',1,'radius',2);
prm=struct('thrr',[-1 1]/5,'reg',.01);
occlPrm=struct('nrows',3,'ncols',3,'nzones',1,'Stot',3,'th',.5);
regPrm = struct('type',1,'K',K,'occlPrm',occlPrm,...
    'loss','L2','R',0,'M',5,'model',model,'prm',prm);
prunePrm=struct('prune',1,'maxIter',2,'th',0.15,'tIni',10);
% I = getImage(conf,imgData.imageID);
% I = cropper(I,imgData.raw_faceDetections);
%fra_db = s40_fra_faces_d;
%%
% for iImg =  1749:length(fra_db)
for iImg =  1:length(fra_db)
    
    for u = 1:1  
        t
        curImgData = fra_db(iImg);
        if (curImgData.classID ~= conf.class_enum.DRINKING)
            continue
        end
        detections = curImgData.raw_faceDetections.boxes(1,:);
        if (detections(end) < 0),continue,end
        conf.get_full_image = false;
        I_orig = getImage(conf,curImgData);
        faceBox = detections(1:4);
        faceBox = inflatebbox(faceBox,1,'both',false);
        I = cropper(I_orig,round(faceBox));
        I =imResample(I,1,'bilinear');
        %     p=shapeGt('initTest',IsT,bboxesT,model,pStar,pGtN,RT1);
        %Create test struct
        %     ff = faceBox;        
        I=  flip_image(I);
        I = imrotate(I,0,'bilinear','loose');
        ff = [1 1 fliplr(size2(I)-1)];
        %     ff(3:4) = ff(3:4)-ff(1:2);
        %     testPrm = struct('RT1',RT1,'pInit',ff,...
        %     'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
        %     'verbose',1);
        %
        RT1 = 20;
        frontal = true;
        if (frontal)
            p=shapeGt('initTest',{I},ff,regModel_frontal.model,...
                regModel_frontal.pStar,regModel_frontal.pGtN,RT1);
            testPrm = struct('RT1',RT1,'pInit',ff,...
                'regPrm',regPrm,'initData',p,'prunePrm',prunePrm,...
                'verbose',1);
            t=clock;[p,pRT] = rcprTest({I},regModel_frontal,testPrm);t=etime(clock,t);
        else
            %         I_orig  = flip_image(I_orig);
            %         ff = flip_box(faceBox,size2(I_orig));
            %         ff(3:4) = ff(3:4)-ff(1:2);
            p_init=shapeGt('initTest',{I},ff,regModel_profile.model,...
                regModel_profile.pStar,regModel_profile.pGtN,RT1);
            testPrm = struct('RT1',RT1,'pInit',ff,...
                'regPrm',regPrm,'initData',p_init,'prunePrm',prunePrm,...
                'verbose',1);
            t=clock;[p,pRT] = rcprTest({I},regModel_profile,testPrm);t=etime(clock,t);
        end
        clf; imagesc2(I);
        plot(p(1:end/2),p(end/2+1:end),'r+');
        %     plotBoxes(flip_box(faceBox,size2(I_orig)));
        pause
    end
end

