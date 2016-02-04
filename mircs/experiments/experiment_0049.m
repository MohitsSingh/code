%% Experiment 0049 %%%%%
%% 8/9/2014
% Make my own facial landmark detector.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    %wSize = 96;
    wSize = 64;
    extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
    initialized = true;
    requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta10'));
end

if (~exist('initialized2','var'))
    myDataPath = '~/storage/misc/kp_pred_data.mat';
    if (exist(myDataPath,'file'))
        load(myDataPath);
    else
        [paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
        L_pts = load('~/storage/data/ptsData');
        ptsData = L_pts.ptsData(1:end);
        poses = L_pts.poses(1:end);
        requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
        all_kp_predictions_local = zeros(length(fra_db),length(requiredKeypoints),5);
        all_kp_predictions_global = zeros(length(fra_db),length(requiredKeypoints),5);
        %
        rolls = [poses.roll];
        pitches = [poses.pitch];
        yaws = [poses.yaw];
        goods = abs(rolls) < 30*pi/180;
        [u,iu] = sort(rolls,'descend');
        edges = [0 20 45 90];
        [b,ib] = histc(180*abs(yaws)/pi,edges);
        poseMap = [90 -90 30 -30 0 0];
        % load the dpm detections on aflw.
        dpmDetsPath = '~/storage/data/aflw_cropped_context/dpm_detections.mat';
        if (exist(dpmDetsPath,'file'))
            load(dpmDetsPath);
        else
            ress = zeros(length(paths),6);
            id = ticStatus( 'loading paths', .5);
            for p = 1:length(paths)
                detPath = j2m('~/storage/aflw_faces_baw',paths{p});
                load(detPath);
                nBoxes = size(res.detections.boxes,1);
                if (nBoxes > 0)
                    ress(p,:) = res.detections.boxes(1,:);
                end
                tocStatus(id,p/length(paths));
            end
            save(dpmDetsPath,'ress');
        end
        scores = ress(:,end);
        %
        bad_imgs = false(size(paths));
        id = ticStatus( 'cropping imgs', .5);
        ims = {};
        pts = {};
        for t = 1:length(paths)
            curBox = round(ress(t,:));
            % make sure all keypoints are inside face detection.
            
            boxToCheck = inflatebbox(curBox(1:4),1.3,'both',false);
            nOutOfBox = ~inBox( boxToCheck, ptsData(t).pts);
            nOutOfBox = nnz(nOutOfBox)/length(nOutOfBox);
            if (nOutOfBox > .8)
                t
                clf; imagesc2(imread(paths{t})); plotBoxes(boxToCheck,'Color','r','LineWidth',2);
                plotPolygons(ptsData(t).pts,'g.');
                drawnow
                %         pause
                bad_imgs(t) = true;
                continue
            end
            %     continue
            ims{t} = cropper(imread(paths{t}),curBox);
            tocStatus(id,t/length(paths));
        end
        
        %
        scores = ress(~bad_imgs,end);
        ims = ims(~bad_imgs);
        ptsData = ptsData(~bad_imgs);
        ress(bad_imgs,:) = [];
        yaws(bad_imgs) = [];
        pitches(bad_imgs) = [];
        goodPaths = paths;goodPaths(bad_imgs) = [];
        %
        T_score = 2.45; % minimal face detection score...
        im_subset = row(find(scores > T_score));
        im_subset = vl_colsubset(im_subset,10000,'random');
        %
        curImgs = ims(im_subset);
        curYaws = 180*yaws(im_subset)/pi;
        curPitches = 180*pitches(im_subset)/pi;
        XX = extractHogsHelper(curImgs);
        XX = cat(2,XX{:});
        save(myDataPath,'XX','wSize','curImgs', 'curYaws', 'curPitches', 'ress','ptsData','im_subset','-v7.3');
    end
    
    initialized2 = true;
    kdtree = vl_kdtreebuild(XX);    
end
%%
debug_ = true;
if debug_
    figure(1);
end
%
outDir = '~/storage/all_kp_preds_new';
ensuredir(outDir);

kpParams.debug_ = debug_;
kpParams.wSize = wSize;
kpParams.extractHogsHelper = extractHogsHelper;
kpParams.im_subset = im_subset;
kpParams.requiredKeypoints = requiredKeypoints;

%%
% trying for stanford_40!!
%%
s40_fra = struct('imageID',{},'imageIndex',{},'isTrain',{},'faceBox',{});
newImageData = augmentImageData(conf,[]);
%%
outDir = '~/storage/s40_kp_preds_new';
ensuredir(outDir);
kpParams.debug_ = true;
for t = 1:length(newImageData)
    t
    R = j2m('~/storage/s40_faces_baw',newImageData(t).imageID);
    curOutPath = j2m(outDir,newImageData(t).imageID);
    if (~debug_ && exist(curOutPath,'file'))
        continue;
    end
    if (~exist(R,'file')),continue,end
    load(R);
    %detections = res.detections;
    
    s40_fra(t).imageID = newImageData(t).imageID;
    s40_fra(t).faceBox = detections(3).boxes(1,1:4);
    conf.get_full_image = false;
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    conf.get_full_image = true;
    [rois,roiBox,I] = get_rois_fra(conf,s40_fra(t),roiParams);
    faceBox = rois(1).bbox;
    bb = round(faceBox(1,1:4));
            
    I = imread('/net/mraid11/export/data/beny/visualTuring/preliminary/indoors09.jpg');
    imshow(I);
    
%     x2(I);
%     bb = getSingleRect(true);
%     bb = inflatebbox(bb,1.3,'both',false);
%     x2(I); plotBoxes(bb);
%     
    I = imcrop(I,makeSquare(bb,true));
    I = imcrop(I);
    I = imResample(I,[192 192],'bilinear');
    bb = [1 1 192 192];
    
    [kp_global,kp_local] = myFindFacialKeyPoints(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams);
    
    p_g = boxCenters(kp_global(:,1:4));
    p_l = boxCenters(kp_local(:,1:4));
    
    [r,ir] = sort(kp_local(:,end),'descend');
    for it = 1:length(ir)
        t = ir(it);
        clf; imagesc2(I); plotPolygons(p_l(t,:),'r+');
        disp(r(it))
        dpc
    end
    
    
    
    
    %         
    s40_fra(t).kp_global = kp_global;
    s40_fra(t).kp_local = kp_local;
end

%%
for t = 1:length(fra_db) % 319
    curOutPath = j2m(outDir,fra_db(t));
    if (~debug_ && exist(curOutPath,'file'))
        continue;
    end
    if (~fra_db(t).isTrain),continue,end
    [I,face_boxes] = getFaceDetectionFRA_DB(conf,fra_db(t));
    %     x2(I); plotBoxes(face_boxes);
    bb = round(face_boxes(1,1:4));
    [kp_global,kp_local] = myFindFacialKeyPoints(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams);
    all_kp_predictions_global(t,:,:) = kp_global;
    all_kp_predictions_kp_local(t,:,:) = kp_local;
end

%% test on rcpr
L = load('/home/amirro/code/3rdparty/rcpr_v1/data/COFW_test.mat');
outDir = '~/storage/s40_kp_preds_new';
ensuredir(outDir);
kpParams.debug_ = true;
%%
conf_ = conf;
cd /home/amirro/code/3rdparty/voc-release5
startup
load ~/code/3rdparty/dpm_baseline.mat
res.model = model;
addpath('~/code/mircs');
%%
for t = 8:length(L.IsT)
    t
    %     R = j2m('~/storage/s40_faces_baw',newImageData(t).imageID);
    %     curOutPath = j2m(outDir,newImageData(t).imageID);
    %     if (~debug_ && exist(curOutPath,'file'))
    %         continue;
    %     end
    %     if (~exist(R,'file')),continue,end
    %     load(R);
    %     detections = res.detections;
    %     s40_fra(t).imageID = newImageData(t).imageID;
    %     s40_fra(t).faceBox = detections.boxes(1,1:4);
    %     conf.get_full_image = false;
    %     roiParams.centerOnMouth = false;
    %     roiParams.infScale = 1.5;
    %     roiParams.absScale = 192;
    %     [rois,roiBox,I] = get_rois_fra(conf,s40_fra(t),roiParams);
            
    I = L.IsT{t};    
    faceBox = L.bboxesT(t,:);
    faceBox([3 4]) = faceBox([3 4])+faceBox([1 2]);
    faceBox = round(inflatebbox(makeSquare(faceBox),[1.5 1.5],'both',false));
    I = cropper(I,faceBox);
        
    [ds, bs] = imgdetect(I, model,model.thresh-.5);
    top = nms(ds, 0.1);           
    faceBox = ds(top(1),:);    
    x2(I); plotBoxes(faceBox);
    %faceBox = rois(1).bbox;
    bb = round(faceBox(1,1:4));
    if (length(size(I))==2)
        I = repmat(I,[1 1 3]);
    end    
    [kp_global,kp_local] = myFindFacialKeyPoints(conf_,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams);
    close all;
    %     s40_fra(t).kp_global = kp_global;
    %     s40_fra(t).kp_local = kp_local;
end

%% trying with a smaller subset
kpParams = kp_params_orig;

sel_ = 1:1000;
% sel_ = 1:length(curImgs);
kpParams.im_subset = kpParams.im_subset(sel_);
my_curImgs = curImgs(sel_);

wSize = 128;
kpParams.wSize = wSize;
extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
kpParams.extractHogsHelper = extractHogsHelper;
XX = extractHogsHelper(my_curImgs);
XX = cat(2,XX{:});


kdtree = vl_kdtreebuild(XX);
outDir = '~/storage/s40_kp_preds_new';
ensuredir(outDir);
kpParams.debug_ = true;
%%
for t = 1:length(newImageData)
    t
    R = j2m('~/storage/s40_faces_baw',newImageData(t).imageID);
    curOutPath = j2m(outDir,newImageData(t).imageID);
    if (~debug_ && exist(curOutPath,'file'))
        continue;
    end
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    s40_fra(t).imageID = newImageData(t).imageID;
    s40_fra(t).faceBox = detections.boxes(1,1:4);
    conf.get_full_image = false;
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    [rois,roiBox,I] = get_rois_fra(conf,s40_fra(t),roiParams);
    faceBox = rois(1).bbox;
    bb = round(faceBox(1,1:4));
    [kp_global] = myFindFacialKeyPoints_2(conf,I,bb,XX,kdtree,my_curImgs,ress,ptsData,kpParams,[]);
end

%%  trying with neural net features....

% run the CNN
% kp_params_orig = kpParams;
kpParams = kp_params_orig;
sel_ = 1:1000;
% sel_ = 1:length(curImgs);
kpParams.im_subset = kpParams.im_subset(sel_);
my_curImgs = curImgs(sel_);

imo = cnn_imagenet_get_batch(my_curImgs, 'averageImage',net.normalization.averageImage,...
    'border',net.normalization.border,'keepAspect',net.normalization.keepAspect,...    
    'numThreads', 1, ...
    'prefetch', false,...
    'augmentation', 'none','imageSize',net.normalization.imageSize);

net_res = vl_simplenn(net, imo);
XX = double(squeeze((net_res(17).x)));  % fc7
XX = normalize_vec(XX);

kdtree = vl_kdtreebuild(XX);
outDir = '~/storage/s40_kp_preds_new';
ensuredir(outDir);
kpParams.debug_ = true;
%%
for t = 1:length(newImageData)
    t
    R = j2m('~/storage/s40_faces_baw',newImageData(t).imageID);
    curOutPath = j2m(outDir,newImageData(t).imageID);
    if (~debug_ && exist(curOutPath,'file'))
        continue;
    end
    if (~exist(R,'file')),continue,end
    load(R);
    detections = res.detections;
    s40_fra(t).imageID = newImageData(t).imageID;
    s40_fra(t).faceBox = detections.boxes(1,1:4);
    conf.get_full_image = false;
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    roiParams.absScale = 192;
    [rois,roiBox,I] = get_rois_fra(conf,s40_fra(t),roiParams);
    faceBox = rois(1).bbox;
    bb = round(faceBox(1,1:4));
    [kp_global] = myFindFacialKeyPoints_2(conf,I,bb,XX,kdtree,my_curImgs,ress,ptsData,kpParams,net);
end

%% cofw...
load /home/amirro/code/3rdparty/voc-release5/cofw_dets.mat
R = load('/home/amirro/code/3rdparty/rcpr_v1/data/COFW_test.mat');
%%

addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');


% train keypoints regressors:
myImgs = cellfun2(@(x) imResample(x,[wSize wSize]),curImgs);
dd = 1;
max_dd = 6;

% randomly sample patches from many parts of the face, and in each face
% remember the offset to each of the keypoints

kps = getKPCoordinates(ptsData(im_subset),ress(im_subset,:)-1,requiredKeypoints);
a = BoxSize(ress(im_subset,:));
f = wSize./a;
for ikp = 1:size(kps,1)
    kps(ikp,:,:) = f(ikp)*kps(ikp,:,:);
end

nPtsPerImage = 100;
margin = 5;
%curPatches = {};
all_feats = {};
all_offsets = {};
toShowStuff = false;
smallPatchSize = [32 32];
smallCellSize = 4;

for u = 7000:10:length(myImgs)
    u
    % sample randomally from this image            
    r = randi([smallPatchSize(1)/2+1, 64-smallPatchSize(1)/2+1],nPtsPerImage,2);    
    r = unique(r,'rows');
    cur_kps = squeeze(kps(u,:,:));
    if (toShowStuff)
        clf; imagesc2(myImgs{u}); plotPolygons(cur_kps,'g.');
        showCoords(r);
    end
    r_boxes = inflatebbox([r r],smallPatchSize,'both',true);
    r_patches = multiCrop2(myImgs{u},r_boxes);
    features = cellfun2(@(x) col(fhog2(im2single(x),smallCellSize)),r_patches);features = double(cat(2,features{:}));
   
    
    cur_offsets = zeros(size(r,1),length(requiredKeypoints),2);
    for ir = 1:size(r,1)
        cur_offsets(ir,:,:) = bsxfun(@minus,cur_kps,r(ir,:));
    end
    
     if (toShowStuff)
        plotBoxes(r_boxes);        
        pause
     end    
    
    all_feats{u} = features;
    all_offsets{u} = cur_offsets;
end


all_feats = double(cat(2,all_feats{:}));
all_offsets = double(cat(1,all_offsets{:}));

nans = (any(any(isnan(all_offsets),3),2));

% fix the nans....

my_offsets = reshape(all_offsets,size(all_offsets,1),[]);

nans = isnan(my_offsets);
nan_patterns = unique(nans,'rows');
nan_patterns(all(nan_patterns==0,2),:) = [];
good_rows = all(nans==0,2);
for iPattern = 1:size(nan_patterns,1)
    iPattern
    cur_p = nan_patterns(iPattern,:);
    needToCorrect = all(repmat(cur_p,size(nans,1),1)==nans,2);    
    neededCoords = cur_p;
    curData = my_offsets(good_rows,:);
    queryData = my_offsets(needToCorrect,~neededCoords);
    D = l2(queryData,curData(:,~neededCoords));
    [m,im] = min(D,[],2);
    corrected_before = my_offsets(needToCorrect,:);
    corrected_after = corrected_before;
    corrected_after(:,neededCoords) = curData(im,neededCoords);    
    my_offsets(needToCorrect,:) = corrected_after;
%     for t = 1:size(corrected_after)
%         clf; hold on; 
%         plot(reshape(corrected_after(t,:),[],2),'g+');
%         plot(reshape(corrected_before(t,:),[],2),'r+');
%         axis image
%         pause
%     end
    
end



all_offsets_n = all_offsets(~nans,:,:);
all_feats_n = all_feats(:,~nans);
kdtree2 = vl_kdtreebuild(all_feats_n);


kdtree2 = vl_kdtreebuild(all_feats);

% train a regressor for the e.g, left eye

regressors = struct;

for iFeat = 1:length(requiredKeypoints)
    regressors(iFeat).offsets = squeeze(all_offsets(:,iFeat,:));
end

for iFeat = 1:length(requiredKeypoints)
    iFeat
    regressors(iFeat).model_x = train(all_offsets(:,iFeat,1), sparse(all_feats'), '-s 11');        
    regressors(iFeat).model_y = train(all_offsets(:,iFeat,2), sparse(all_feats'), '-s 11');
end
%%
close all;
kpParams.debug_ = false
cofw_res = struct;
% for t = 1:length(R.IsT)
    for t = 133
    t    
    detections = res(t).detections;
    I = R.IsT{t}; 
    if (size(I,3)==1)
        I = cat(3,I,I,I);
    end
    faceBox = detections.boxes(1,1:4);
    bb = round(faceBox(1,1:4));
    kpParams.debug_ =true;
    figure(1);
    %[cofw_res(t).kp_global,cofw_res(t).kp_local] = myFindFacialKeyPoints_new(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams,reshape(my_offsets,size(all_offsets)),kdtree2,all_feats);
    [cofw_res(t).kp_global,cofw_res(t).kp_local] = myFindFacialKeyPoints_new(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams,all_offsets_n,kdtree2,all_feats_n);
    
%     [cofw_res(t).kp_global,cofw_res(t).kp_local] = myFindFacialKeyPoints(conf,I,bb,XX,kdtree,curImgs,ress,ptsData,kpParams);
    
end
%%
save cofw_my_landmarks cofw_res

