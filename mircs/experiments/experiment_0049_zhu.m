%% Experiment 0049 %%%%%
%% 8/9/2014
% Create a model for the probability of a face patch, in order to find
% what are object that don't "belong" to the face.

if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    roiParams.infScale = 3.5;
    roiParams.absScale = 200*roiParams.infScale/2.5;
    normalizations = {'none','Normalized','SquareRoot','Improved'};
    addpath(genpath('~/code/3rdparty/SelectiveSearchCodeIJCV/'));
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    featureExtractor = learnParams.featureExtractors{1};
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    initialized = true;
    
    addpath('/home/amirro/code/3rdparty/dsp-code');
    load ct101_pca_basis.mat pca_basis 
end

load  ~/storage/misc/zhu_aflw_landmarks.mat imgPaths all_lm_data

allScores = [all_lm_data.s];
goods = allScores>0;
all_lm_data = all_lm_data(goods);
imgPaths = imgPaths(goods);
ress = zeros(length(imgPaths),6);
id = ticStatus( 'loading imgPaths', .5);

for p = 1:length(imgPaths)
    detPath = j2m('~/storage/aflw_faces_baw',imgPaths{p});
    load(detPath);
    ress(p,:) = res.detections.boxes;
    tocStatus(id,p/length(imgPaths));
end

scores = ress(:,end);
% [scores,iscores] = sort(ress(:,end),'ascend');
% figure(1);displayImageSeries(conf,imgPaths(iscores(1:100:end)),.1)
bad_imgs = false(size(imgPaths));
id = ticStatus( 'cropping imgs', .5);
ims = {};
for t = 1:length(imgPaths)
    curBox = round(ress(t,:));
    % make sure all keypoints are inside face detection.    
    ims{t} = cropper(imread(imgPaths{t}),curBox);
    if (boxesOverlap([50 50 200 200],ress(t,1:4)) < .3)
        bad_imgs(t) = true;
        clf; imagesc2(imread(imgPaths{t})); plotBoxes(ress(t,1:4),'Color','r','LineWidth',2);
        %             pause
        continue;
    end
    %         clf; imagesc2(imread(imgPaths{t})); plotBoxes(ress(t,1:4));
    %         pause
    
    tocStatus(id,t/length(imgPaths));
end
nKP = 68;
all_kp_predictions_local = zeros(length(fra_db),nKP,5);
all_kp_predictions_global = zeros(length(fra_db),nKP,5);

requiredKeypoints = [1:5:68];

%%
scores = ress(~bad_imgs,end);
ims = ims(~bad_imgs);
ress(bad_imgs,:) = [];
all_lm_data(bad_imgs) = [];
%%
T_score = 2.45;
im_subset = row(find(scores > T_score));
im_subset = vl_colsubset(im_subset,1000,'random');
%%
curImgs = ims(im_subset);
wSize = 64;
extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
XX = extractHogsHelper(curImgs);
XX = cat(2,XX{:});
kdtree = vl_kdtreebuild(XX);

%%
%%
% 2. split faces into 3 yaw groups.
% 3. have a keypoint regressor for each yaw group.


%my_requiredKeypoints = {'MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'}



if debug_
    figure(1);
end
%
% ppp = randperm(length(fra_db));
%%
outDir = '~/storage/all_kp_preds';
ensuredir(outDir);
debug_ = true;
for it =1:length(fra_db) % 319
    t = it
    if (fra_db(t).classID~=1),continue,end
    curOutPath = j2m(outDir,fra_db(t));
    if (~debug_ && exist(curOutPath,'file'))
        continue;
    end
    R = j2m('~/storage/fra_faces_baw',fra_db(t));
    load(R);
    res.detections = res.detections(3);
    boxes = cat(1,res.detections.boxes);
    rots = [res.detections.rot];
    [u,iu] = max(boxes(:,end),[],1);
    faceRot = rots(iu);
    faceBox = boxes(iu,:);
%     disp(['cur pose: ' num2str(poseMap(faceBox(5))) ', cur rot: ' num2str(faceRot)]);
    roiParams.absScale = 192;
    faceBox(1:4) = faceBox(1:4)*roiParams.absScale/192;
    bb = round(boxes(iu(1),1:4));
    roiParams.centerOnMouth = false;
    roiParams.infScale = 1.5;
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams); %
    rrr = [.8:.2:1.2];
    [resizeX,resizeY] = meshgrid(rrr,rrr);
%     %
    inflations = [resizeX(:) resizeX(:)];
    inflations = [rrr(:) rrr(:)];
    n = length(size(inflations,1));
    subImg = multiCrop(conf,I,bb);
    knn = 1;
    knn = round(knn^.5)^2;
    x = extractHogsHelper(subImg);
    x = cat(2,x{:});
    [id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*100);
    [s,is] = min(sum(dists,1));
    
    neighborInds = im_subset(id(:,is));
    %kps = getKPCoordinates(ptsData(neighborInds),ress(neighborInds,:)-1,requiredKeypoints);
    kps = getKPCoordinates_zhu(all_lm_data(neighborInds),ress(neighborInds,:)-1);
    
    a = BoxSize(ress(neighborInds,:));
    f = wSize./a;
    for ikp = 1:size(kps,1)
        kps(ikp,:,:) = f(ikp)*kps(ikp,:,:);
    end
    pMap = zeros(wSize);
    % create a prediction for the points using the nearest neighbors.
    myImgs = curImgs(id(:,is));
    myImgs = cellfun2(@(x) imResample(x,[wSize wSize]),myImgs);
    
    boxFrom = bb(is,:);
    boxTo = [1 1 fliplr(size2(pMap))];
    conf.detection.params.detect_min_scale = .6;
    I2 = imResample(I,1.9*wSize/size(I,1),'bilinear');
    ss = wSize/8;
    conf.features.winsize = [ss ss];
    II = imResample(subImg{is},[wSize,wSize]);
    T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
    for ptsType = requiredKeypoints
        curPts = reshape(kps(:,ptsType,:),[],2);
        bad_pts = (any(isnan(curPts),2));
        bad_pts = bad_pts | any(curPts < 1,2);
        curPts(bad_pts,:) = [];
        if (none(curPts))
            all_kp_predictions_local(t,ptsType,:) = -inf;
            continue;
        end
        %         myImgs(bad_pts) = [];
        annParams.maxFeats = 10;
        
                
        [features,offsets] = prepareANNStarData(conf,myImgs(~bad_pts),curPts,annParams);
        kdtree_feats = vl_kdtreebuild(features,'Distance','L2');
        %imagesc2(myImgs{1}); plotPolygons(squeeze(kps(1,:,:)),'g+');
        curResize = 1;%;resizeRatios(iResize);
        flip = false;
        params.rot = rots(iu);
        params.resizeRatio = curResize;
        params.flip = flip;
        
        params.max_nn_checks = 100;
        params.nn = 1;
        params.stepSize = 4;
        
        
        [pp] = predictBoxesANNStar(conf,II,features,offsets,kdtree_feats,params);
%         pp = predictBoxesRegression(conf,II,features,offsets,kdtree_feats,params);
        
        
        pp_I = imtransform(pp,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
        all_kp_predictions_local(t,ptsType,:)=pMapToBoxes(pp_I,5,1);
        pMap=pMap+normalise(pp.^2);
    end
    
    
    
    %     break
    curBoxes = squeeze(all_kp_predictions_local(t,:,:));
    myKPS = kps;
    % make a probability map to start with
    Z = zeros(wSize);
    boxes = zeros(length(requiredKeypoints),5);
    for iReqKeyPoint = 1:length(requiredKeypoints)
        xy = reshape(kps(:,iReqKeyPoint,:),[],2);
        [u,v] = meshgrid(1:size(Z,2),1:size(Z,1));
        xy(any(isnan(xy),2),:) = [];
        dd = l2([u(:) v(:)],xy);
        sig_ = 20;
        dd = reshape(sum(exp(-dd/sig_)/2,2),size(Z));
        dd_I = imtransform(dd,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
        all_kp_predictions_global(t,iReqKeyPoint,:) = pMapToBoxes(dd_I,5,1);
        Z = Z+dd;
    end
    
    
    curKP_global = squeeze(all_kp_predictions_global(t,:,:));
    curKP_local = squeeze(all_kp_predictions_local(t,:,:));
    
    if (~debug_)
        save(curOutPath,'curKP_global','curKP_local');
    end
    
    
    if ~debug_
        continue
    end
    
    % transform back to I's coordinates
    boxFrom = bb(is,:);
    boxTo = [1 1 fliplr(size2(Z))];
    T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
    Z_I = imtransform(Z,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
    
    %     clf;subplot(1,2,1); imagesc2(sc(cat(3,Z,II),'prob'));
    %     subplot(1,2,2); imagesc2(sc(cat(3,res,I),'prob'));
    %     pause;continue;
    %     Z = imfilter(Z,fspecial('gauss',[15 5]));
    %
    %    figure(2); clf;  %
    %
    
    mm = 2; nn = 3;
    figure(1);
    clf;
    vl_tightsubplot(mm,nn,1);
    imagesc2(I); plotBoxes(bb(is,:));
    plotBoxes(faceBox,'r--','LineWidth',2);
    vl_tightsubplot(mm,nn,2);
    imagesc2(imResample(subImg{is},[wSize,wSize]));
% % %     hold on;
% % %     specs = {'c+','m+','y+','k+','bd','r.','g.','cd','md','yd','kd'};
% % %     for iReqKeyPoint = 1:length(requiredKeypoints)
% % %         xy = reshape(kps(:,iReqKeyPoint,:),[],2);
% % %         if (all(isnan(xy(:))))
% % %             xy = [-100 -100];
% % %         end
% % %         plot(xy(:,1),xy(:,2),specs{iReqKeyPoint});
% % %     end
    % %     legend(requiredKeypoints);
    vl_tightsubplot(mm,nn,3);
    M = mImage(myImgs);
    imagesc2(M);
    % %
    
    %     vl_tightsubplot(mm,nn,4);
    %     plot(aspects(:),sum(dists),'r+');% hold on; plot(is,s,'m*');
    
    if (is==1)
        plot(is,s,'gs');
    end
    vl_tightsubplot(mm,nn,5);
    %   imagesc2(sc(cat(3,Z,im2double(II)),'prob'));
    imagesc2(I);
    plotBoxes(squeeze(all_kp_predictions_global(t,:,:)));
    
    vl_tightsubplot(mm,nn,4);
    imagesc2(sc(cat(3,Z,im2double(II)),'prob'));
    
    vl_tightsubplot(mm,nn,6);
    imagesc2(I);
    %     for zz = 1:length(curBoxes)
    
    %     end
    %         imagesc2(sc(cat(3,pMap,II),'prob'));
    %plotBoxes(curBoxes(zz,:),specs{zz});
    plotBoxes(curBoxes);
    
    %      figure(2);clf;
    %      vl_tightsubplot(2,1,1);imagesc2(II);
    %      vl_tightsubplot(2,1,2);imagesc2(II1);        
    pause
    %     allPredictions{t,iPredictor} = pMap;
end
%
save allKPPreds.mat all_kp_predictions_global all_kp_predictions_local

%ed%
%predictionResults

%requiredKeypoints = {'MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};%,'LeftEyeCenter','RightEyeCenter'};


requiredKeypoints = {'MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter','LeftEyeCenter','RightEyeCenter',...
    'RightEyeRightCorner','RightEyeLeftCorner','LeftEyeRightCorner','LeftEyeLeftCorner'};

% % % % % predictors = struct('pose',{},'keypointName',{},'features',{},'offsets',{},'kdtree',{});
% % % % % resizeFactor = .5;
% % % % % close all
% % % % % debug_ = true;
% % % % % iPredictor = 0;
% % % % % pathOrder = randperm(length(imgPaths));
% % % % % for iPose = 1:3
% % % % %     pose_sel = iPose;
% % % % %     for iReqKeyPoint = 1:length(requiredKeypoints)
% % % % %         %
% % % % %         imgs = {};
% % % % %         pts = {};
% % % % %         tt = 0;
% % % % %         figure(1)
% % % % %         
% % % % %         maxImgs = 100;
% % % % %         
% % % % %         for it = 1:1:length(imgPaths)
% % % % %             it
% % % % %             t = pathOrder(it);
% % % % %             if (goods(t) && ib(t)==pose_sel)
% % % % %                 it
% % % % %                 ii = strcmp(requiredKeypoints{iReqKeyPoint}, `(t).pointNames);
% % % % %                 if (~any(ii))
% % % % %                     continue;
% % % % %                 end
% % % % %                 curIm = imread(imgPaths{t});
% % % % %                 curPts = ptsData(t).pts;
% % % % %                 curIm = imResample(curIm,resizeFactor);
% % % % %                 curPts = curPts*resizeFactor;
% % % % %                 curPts = curPts(ii,:);
% % % % %                 imgs{end+1} = curIm;
% % % % %                 pts{end+1} = curPts;
% % % % %                 tt = tt+1;
% % % % %                 if (tt >= maxImgs)
% % % % %                     break;
% % % % %                 end
% % % % %                 if (debug_)
% % % % %                     
% % % % %                     if (mod(tt,10)==0)
% % % % %                         clf;imagesc2(curIm);
% % % % %                         plotPolygons(curPts,'g.');
% % % % %                         drawnow
% % % % %                     end
% % % % %                 end
% % % % %             end
% % % % %         end
% % % % %         %
% % % % %         [features,offsets] = prepareANNStarData(conf,imgs,pts);
% % % % %         kdtree = vl_kdtreebuild(features,'Distance','L2');
% % % % %         iPredictor = iPredictor + 1;
% % % % %         predictors(iPredictor) = struct('pose',pose_sel,'keypointName',requiredKeypoints{iReqKeyPoint},...
% % % % %             'features',features,'offsets',offsets,'kdtree',kdtree);
% % % % %     end
% % % % % end
% % % % % 
% % % % % 
% % % % % 


%%

% allPredictions = {};
p = 1:length(fra_db);
params.nn = 1;
params.stepSize = 1;
params.max_nn_checks = 100;
debug_ = true;
% profile off
figure(1)
for iPredictor = 1:length(predictors)
    
    features = predictors(iPredictor).features;
    offsets = predictors(iPredictor).offsets;
    kdtree = predictors(iPredictor).kdtree;
    
    if (predictors(iPredictor).pose~=2)
        continue;
    end
    predictors(iPredictor)
    
    for it = 1:1:length(fra_db)
        
        it = 319
        t = p(it)
        roiParams.infScale = 1.7;
        roiParams.absScale = 145;
        roiParams.centerOnMouth = false;
        bestRot = 0;
        bestFlip = 0;
        bestResizeRatio =0;
        bestScore = -inf;
        %     resizeRatios = .8:.1:1.2;
        resizeRatios = 1;
        %for rots = -10:10:10
        % %         for rots = 0
        % %             for flip = 0%[0 1]
        % %                 for iResize = 1:length(resizeRatios)
        % %                     curResize = resizeRatios(iResize);
        % %                     params.rot = rots;
        % %                     params.resizeRatio = curResize;
        % %                     params.flip = flip;
        % %                     [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
        % %                     %     figure,imagesc2(I)
        % %                     [pMap] = predictBoxesANNStar(conf,I,features,offsets,kdtree,params);
        % %
        % %                     curScore = max(pMap(:));
        % %                     if (curScore > bestScore)
        % %                         bestScore = curScore;
        % %                         bestRot = rots;
        % %                         bestFlip = flip;
        % %                         bestResizeRatio = curResize;
        % %                         if (debug_)
        % %                             clf; subplot(1,2,1); imagesc2(I);
        % %                             subplot(1,2,2); imagesc2(sc(cat(3,pMap.^2,I),'prob'));
        % %                             title(num2str(curScore));
        % %                             drawnow
        % %                         end
        % %                     end
        % %                     %             pause(.1)
        % %                 end
        % %             end
        % %         end
        params.rot = bestRot;
        params.flip = bestFlip;
        params.resizeRatio = bestResizeRatio;
        
        
        params.rot = 0;
        params.flip = 0;
        params.resizeRatio = 1;
        
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
        [pMap] = predictBoxesANNStar(conf,I,features,offsets,kdtree,params);
        if (debug_)
            clf; subplot(1,2,1); imagesc2(I);
            subplot(1,2,2); imagesc2(sc(cat(3,pMap.^2,I),'prob'));
            pause
        else
            allPredictions{t,iPredictor} = pMapToBoxes(pMap,5,3);
        end
        %     allPredictions{t,iPredictor} = pMap;
    end
end

% save allPredictions allPredictions


%% show some results
p = 1:length(fra_db);

debug_ = true;
% profile off
figure(1)

kp = {'MouthLeftCorner','MouthCenter','MouthRightCorner','ChinCenter','NoseCenter'};
predNames = {predictors.keypointName};

sels = {};
for m = 1:length(kp)
    sels{m} = strmatch(kp{m},predNames);
end

xys = {};
u = cellfun2(@(x) x(1,end), allPredictions(:,6:10));
u = cellfun(@(x) x ,u);
min_u = min(u);
max_u = max(u);


% logistic regression : transform from range to 0, 1
% [m,xo] = hist(u(:,1));

% [a,b] = histc(u(:,1),[xo]);


% for iPredictor = 1:length(predictors)
for it = 319:3:length(fra_db)
    t = p(it)
    roiParams.infScale = 1.7;
    roiParams.absScale = 145;
    roiParams.centerOnMouth = false;
    bestRot = 0;
    bestFlip = 0;
    bestResizeRatio =0;
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),roiParams);
    
    xy = zeros(5,5);
    
    for iSel = 1:length(sels)
        curPredictions = allPredictions(t,sels{iSel});
        curPredictions = curPredictions(2);
        curPredictions = cellfun2(@(x) x(1,:),curPredictions);
        curPredictions = cat(1,curPredictions{:});
        [b,ib] = max(curPredictions(:,end));
        xy(iSel,:) = (curPredictions(ib,1:5));
    end
    %         curPredictions = curPredictions(ib,:);
    if (debug_)
        clf; imagesc2(I);
        plotBoxes(xy);
        bc = boxCenters(xy);
        plot(bc(1:3,1),bc(1:3,2),'m-','LineWidth',2);
        plotBoxes(xy(5,:),'c','LineWidth',2);
        plotBoxes(xy(4,:),'r','LineWidth',2);
        
        (xy(:,end)'-min_u)./max_u
        
        %(xy(:,end)'-min_u)./max_u
        
        pause
    end
    %         allPredictions{t,iPredictor} = pMapToBoxes(pMap,5,3);
    %     allPredictions{t,iPredictor} = pMap;
end
% end



%%
%     I = imread(imgPaths{t});
%     I = imResample(I,[120 120]);
conf.detection.params.detect_min_scale = 1;
opts.show = false;
maxImageSize = 300;
opts.maxImageSize = maxImageSize;
spSize = 10;
opts.pixNumInSP = spSize;
conf.get_full_image = true;
[sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I),opts);

clf; subplot(1,2,1);
imagesc2(I);
subplot(1,2,2);
imagesc2(sc(cat(3,sal,I),'prob_jet'));
pause;continue


%     figure,imagesc2(sal)
%     figure,imagesc2(I)
%
%     I = I(1:end/2,:,:);
%     [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,im2single(curIm));
[F,X] = vl_phow(im2single(I),'Step',1,'FloatDescriptors','true','Fast',true,'Sizes',[6],'Color','gray');
%     [X,uus,vvs,scales,~,boxes ] = allFeatures( conf,im2single(I));
%     F = boxCenters(boxes)';
bads = sum(X)==0;
F(:,bads) = [];
X(:,bads) = [];
X = rootsift(X);
nn = 1;
nChecks = 1000;
[ind_all,dist_all] = vl_kdtreequery(kdtree,xx,X,'numneighbors',nn,'MaxNumComparisons',nChecks);

%     f = sub2ind2(size2(I),');
xy = round(F([2 1],:))';
dist_all = dist_all';
dist_all = dist_all(:);
xy = repmat(xy,length(dist_all)/size(xy,1),1);
%Z = accumarray(xy,exp(-dist_all),size2(I));
Z = accumarray(xy,dist_all.^2,size2(I));
%     Z = imdilate(Z,ones(4));
clf;
subplot(2,1,1); imagesc2(I);
subplot(2,1,2);imagesc2(Z);
%subplot(2,1,2); imagesc2(sc(cat(3,Z.^2,I),'prob_jet'))
pause



