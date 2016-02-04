%% Experiment 0049 %%%%%
%% 8/9/2014
% Create a model for the probability of a face patch, in order to find
% what are object that don't "belong" to the face.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
    initialized = true;
end

dataPath = '~/storage/misc/landmarkDetector.mat';

if (exist(dataPath,'file'))
    load(dataPath);
else        
    [paths,names] = getAllFiles('~/storage/data/aflw_cropped_context','.jpg');
    L_pts = load('~/storage/data/ptsData');
    ptsData = L_pts.ptsData(1:end);
    poses = L_pts.poses(1:end);
    requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};    
    requiredKeypoints = {'LeftEyeLeftCorner','LeftEyeRightCorner','RightEyeLeftCorner','RightEyeRightCorner','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};    
    all_kp_predictions_local = zeros(length(fra_db),length(requiredKeypoints),5);
    all_kp_predictions_global = zeros(length(fra_db),length(requiredKeypoints),5);    
    %%
    rolls = [poses.roll];
    pitches = [poses.pitch];
    yaws = [poses.yaw];
    
    %goods = abs(rolls) < 10*pi/180;
    goods = abs(rolls) < 30*pi/180;
    [u,iu] = sort(rolls,'descend');
    % figure,hist(abs(180*yaws(goods)/pi),20)
    edges = [0 20 45 90];
    [b,ib] = histc(180*abs(yaws)/pi,edges);
    
    poseMap = [90 -90 30 -30 0 0];
    
    %% load the dpm detections on aflw.
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
    %%
    bad_imgs = false(size(paths));
    id = ticStatus( 'cropping imgs', .5);
    ims = {};
    for t = 4836:length(paths)
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
    
    %%
    scores = ress(~bad_imgs,end);
    ims = ims(~bad_imgs);
    ptsData = ptsData(~bad_imgs);
    ress(bad_imgs,:) = [];
    yaws(bad_imgs) = [];
    pitches(bad_imgs) = [];
    goodPaths = paths;goodPaths(bad_imgs) = [];
    
    save('~/storage/misc/landmarkDetector.mat');
end

%%
T_score = 2; % minimal face detection score...
im_subset = row(find(scores > T_score));
im_subset = vl_colsubset(im_subset,inf,'random');
%%
curImgs = ims(im_subset);
curYaws = 180*yaws(im_subset)/pi;
curPitches = 180*pitches(im_subset)/pi;
wSize = 96;
extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[wSize wSize],'bilinear')))) , y);
%X = cellfun2(@(x) col(fhog2(im2single(imResample(x,[80 80],'bilinear')))),curImgs);
XX = extractHogsHelper(curImgs);
XX = cat(2,XX{:});
kdtree = vl_kdtreebuild(XX);
disp('--------------------------------------------------finished setting up------------');

%%
debug_ = true;
if debug_
    figure(1);
end
%

I = [];
detectionBox = [];

%% out of scratch  :-)
%
bb = '~/data/afw/';
d = dir(fullfile(bb,'*.jpg'));

ress1 = {};
fileNames = {};
for u = 1:length(d)
    u
    R = j2m('~/storage/afw_faces_baw',d(u).name);
    load(R);
    detections = res.detections;
    fileNames{end+1} = fullfile(bb,d(u).name);
    %     clf; imagesc2(imread(fullfile(bb,d(u).name)));
    %     detections = detections(3);
    if (isempty(detections.boxes))
        ress1{end+1} = zeros(1,5);
    else
        ress1{end+1} = detections.boxes(1,[1:4 6]);
    end
    %     plotBoxes(detections.boxes(1,:));
    %     drawnow
    %     pause(.1)
end

figure(1)
rr = cat(1,ress1{:});
[r1,ir1] = sort(rr(:,end),'descend');
% for it = 1:1:length(r)
%     
%     ir(1)
%     
%     clf; imagesc2(imread(fileNames{ir(it)}));
%     plotBoxes(rr(ir(it),:));
%     drawnow
%     pause(.1)
% end

%%

doAnn = true;
for it = 153:1:200
    it
    
    %kk = ir1(it);
    kk = it
    fileNames{kk}
    I = imread(fileNames{kk});
    detectionBox = rr(kk,:);    
    detectionBox1 =  round(inflatebbox(detectionBox,[2 2],'both',false));
    detectionBox = detectionBox(1:4)-detectionBox1([1 2 1 2]);
    I = cropper(I,detectionBox1);
    figure(2); clf; imagesc2(I);
    %%
    
    bb = detectionBox;
    rotations = -6:3:6;
%     rotations = 0;
    bb = repmat(bb,length(rotations),1);
    bb = rotate_bbs(bb,I,rotations,false);
    bbb = {};
    for inflations = 1
%                     for inflations = .8:.05:1.2
        for ibb = 1:length(bb)
            curBB = bb{ibb};
            [x,y] = inflatePolygon(curBB(:,1),curBB(:,2),inflations);
            bbb{end+1} = [x y];
        end
    end
    bb = bbb;
    
    subImg = {};
    for iBB = 1:length(bb)
        boxFrom = bb{iBB};
        boxTo = [1 1 wSize wSize];
        [subImg{iBB},T]= rectifyWindow(I,boxFrom,wSize([1 1]));
    end
    
    knn = 25;
    x = extractHogsHelper(subImg);
    x = cat(2,x{:});
    knn = round(knn^.5)^2;
    [id,dists] = vl_kdtreequery(kdtree,XX,x,'NUMNEIGHBORS',knn,'MAXNUMCOMPARISONS',knn*1000);
    ids_orig = id;
    dists_orig = dists;
    
    [s,is] = min(sum(dists(:,:),1));
    all_dists(iDet) = sum(dists(:,is));
    
    neighborInds = im_subset(id(:,is));
    kps = getKPCoordinates(ptsData(neighborInds),ress(neighborInds,:)-1,requiredKeypoints);
    k
    a = BoxSize(ress(neighborInds,:));
    f = wSize./a;
    for ikp = 1:size(kps,1)
        kps(ikp,:,:) = f(ikp)*kps(ikp,:,:);
    end
    pMap = zeros(wSize);
    % create a prediction for the points using the nearest neighbors.
    myImgs = curImgs(id(:,is));
    myImgs = cellfun2(@(x) imResample(x,[wSize wSize]),myImgs);
    
    T = cp2tform(box2Pts(boxTo),bb{is},'affine');
    % make a probability map to start with
    Z = zeros(wSize);
    boxes = zeros(length(requiredKeypoints),5);
    for iReqKeyPoint = 1:length(requiredKeypoints)
        xy = reshape(kps(:,iReqKeyPoint,:),[],2);
        [u,v] = meshgrid(1:size(Z,2),1:size(Z,1));
        xy(any(isnan(xy),2),:) = [];
        dd = l2([u(:) v(:)],xy);
        sig_ = 5;
        dd = reshape(sum(exp(-dd/sig_)/2,2),size(Z));
        dd_I = imtransform(dd,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
        all_kp_predictions_global(t,iReqKeyPoint,:) = pMapToBoxes(dd_I,5,1);
        Z = Z+dd;
    end
    
    
    II = imResample(subImg{is},[wSize,wSize]);
    %         T = cp2tform(box2Pts(boxTo),box2Pts(boxFrom),'affine');
    if (doAnn)
        for ptsType = 1:length(requiredKeypoints)
            curPts = reshape(kps(:,ptsType,:),[],2);
            bad_pts = (any(isnan(curPts),2));
            bad_pts = bad_pts | any(curPts < 1,2);
            curPts(bad_pts,:) = [];
            if (none(curPts))
                all_kp_predictions_local(t,ptsType,:) = -inf;
                continue;
            end
            %         myImgs(bad_pts) = [];
            annParams.maxFeats = 100;
            
            
            [features,offsets] = prepareANNStarData(conf,myImgs(~bad_pts),curPts,annParams);
            kdtree_feats = vl_kdtreebuild(features,'Distance','L2');
            %imagesc2(myImgs{1}); plotPolygons(squeeze(kps(1,:,:)),'g+');
            curResize = 1;%;resizeRatios(iResize);
            flip = false;
            params.rot = 0;
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
    end
    
    curKP_global = squeeze(all_kp_predictions_global(t,:,:));
    curKP_local = squeeze(all_kp_predictions_local(t,:,:));
    
    if (~debug_)
        save(curOutPath,'curKP_global','curKP_local');
    end
    
    % transform back to I's coordinates
    boxTo = [1 1 fliplr(size2(Z))];
    T = cp2tform(box2Pts(boxTo),bb{is},'affine');
    Z_I = imtransform(Z,T,'XData',[1 size(I,2)],'YData',[1 size(I,1)],'XYScale',1);
    
    mm = 2; nn = 3;
    figure(1);
    clf;
    vl_tightsubplot(mm,nn,1);
    imagesc2(I); plotPolygons(bb{is},'g-');
    plotBoxes(faceBox,'r--','LineWidth',2);
    vl_tightsubplot(mm,nn,2);
    imagesc2(imResample(subImg{is},[wSize,wSize]));
    vl_tightsubplot(mm,nn,3);
    M = mImage(myImgs);
    imagesc2(M);
    % %
    vl_tightsubplot(mm,nn,5);
    imagesc2(I);
    
    global_pred_boxes = squeeze(all_kp_predictions_global(t,:,:));
    plotBoxes(global_pred_boxes);
    
    xpoly = bb{is};
    bbb1 = inflatebbox(pts2Box(xpoly),[1.3 1.3],'both',false);
    xlim(bbb1([1 3]));
    ylim(bbb1([2 4]));
    
    bbb = boxCenters(global_pred_boxes);
    vl_tightsubplot(mm,nn,2);
    if (doAnn)
    vl_tightsubplot(mm,nn,6);
    imagesc2(I);
        xlim(bbb1([1 3]));
    ylim(bbb1([2 4]));
    plotBoxes(curBoxes);
    end
    % vl_tightsubplot(mm,nn,1);plotPolygons(bb{is},'m--');
    pause
end
%%


