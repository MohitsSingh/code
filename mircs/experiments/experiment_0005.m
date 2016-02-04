%%%%%% Experiment 5 %%%%%%%
% Oct. 28, 2013

% The purpose of this experiment is to extract some more features from the
% face area (contours, etc.) and see how they combine together with the
% saliency measure and landmark localization for the drinking action.
% 10/11/2013: 
% It seems that the discovered features can provide a good cue but
% still, the search doesn't bring up results which seem very relevant.
% There can be several feature types to suppress the wrong results,
% but instead I'm trying to make the search more direct.
% now trying to discover U shapes, which are simple to characterize

echo off;
if (~exist('toStart','var'))
    initpath;
    weightVector = [1 10 10 0*-.01 10 3 1 1 10 1 0];
        
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    addpath('/home/amirro/code/3rdparty/FastLineSegmentIntersection/');
    addpath('/home/amirro/code/3rdparty/PCA_Saliency_CVPR2013 - v2');
    addpath('/home/amirro/code/3rdparty/guy');
    config;
    load ~/storage/misc/imageData_new;
    %imageData = initImageData;
    toStart = 1;
    conf.get_full_image = false;
    imageSet = imageData.train;
    face_comp = [imageSet.faceLandmarks.c]';
    cur_t = imageSet.labels;
    conf.features.vlfeat.cellsize = 8;
    conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
    conf.features.winsize = [8 8];
    conf.detection.params.detect_add_flip = 0;
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    allFeats = cell(size(cur_t));
    iv = 1:length(cur_t);
    facesPath = fullfile('~/mircs/experiments/common/faces_cropped_new.mat');
    load '/home/amirro/mircs/experiments/experiment_0001/sals_new.mat';
    L = load('/home/amirro/mircs/experiments/experiment_0001_improved/exp_result.mat');
    load(facesPath);
    [objectSamples,objectNames] = getGroundTruth(conf,train_ids,train_labels);
    sfe = MaskShapeFeatureExtractor(conf);
    sfe.absoluteFrame = true;
    posemap = [-90:15:90 inf];
    imageNames = {objectSamples.sourceImage};
    objNames = {objectSamples.name};
%     sum((~cellfun(@isempty,strfind(objNames,'cup'))))
    
    comps = [imageData.train.faceLandmarks.c];
    comps(comps==0) = 14;
    
end

%%
% % debug_ = true;
% % conf.get_full_image = 0;
% %
% %
% % % first, do it only for the side cup/bottle. afterwards continue to other
% % % categories.
% %
featureData = struct('img',{},'mask',{},'ucm','metadata');
count_ = 0;
falses = struct('img',{},'mask',{},'ucm','metadata');
falseChance = 0;
falseCount = 0;
%%
debug_ = false;



%%
conf.get_full_image = false;



Rs_train = cell(1,length(imageData.train.imageIDs));
for k =1:length(imageData.train.imageIDs)
    k
    if (~cur_t(k))
        continue;
    end
    if (imageData.train.faceScores(k) < -.7)
        continue;
    end
    Rs_train{k} = extractCandidateFeatures3(conf,imageData.train,sal_train,k,true);
end

save /home/amirro/mircs/experiments/experiment_0005/Rs_train Rs_train
for k =1:length(imageData.test.imageIDs)
    k
    
    Rs_test{k} = extractCandidateFeatures3(conf,imageData.test,sal_test,k,true);   
end

save /home/amirro/mircs/experiments/experiment_0005/Rs_test Rs_test


% U shapes!

for k = 804:length(imageData.train.imageIDs)
%     k = 803
    if (cur_t(k))
        Us_train{k} = extractCandidateFeatures4(conf,imageData.train,sal_train,k,true);
    end
end

%%
%%
doTrain = 1;
if (doTrain)
    imageSet = imageData.train;
    sals = sal_train;
    Rs_t = Rs_train;
    comps = [imageSet.faceLandmarks.c];
    comps(comps==0) = 14;
    T_saliency = L.train_saliency;
    faceSet = faces.train_faces;
    cur_t = imageSet.labels;
else
    imageSet = imageData.test;
    sals = sal_test;
    Rs_t = Rs_test;
    comps = [imageSet.faceLandmarks.c];
    comps(comps==0) = 14;
    T_saliency = L.test_saliency;
    faceSet = faces.test_faces;
    cur_t = imageSet.labels;
end

sel_ = cell2mat(cellfun2(@(x)~isempty(x),Rs_t));

cur_rs = Rs_t(sel_);
curScores = zeros(size(cur_rs));
for k = 1:length(curScores)
    r = cur_rs{k};
    if (isempty(r) || isempty(r.bbox))
        curScores = -inf;
        continue;
    end
    c_score = exp(-(r.bc(:,1)-.5).^2*10);
    horz_extent = r.horz_extent;
    ucmStrengths = r.ucmStrength;
    salStrengths = r.salStrength/255;
    isConvex = double(r.isConvex');
    bbox = r.bbox;
    
    % score by the overlap of the bbox with the mouth bbox.
    mouthBox = imageSet.lipBoxes(k,:);
    faceBox =  imageSet.faceBoxes(k,:);
    fSize = faceBox(3)-faceBox(1)+1;
    mouthBox = mouthBox-faceBox([1 2 1 2]);
    mouthBox = mouthBox/fSize;
    
%     
%     ovp = boxesOverlap(bbox,mouthBox);
    
    
    
    
    y_top = bbox(:,2);
    ss = c_score+1*(horz_extent)+ucmStrengths+salStrengths;
%     ss = ss+(ovp>0.2);
%     ss = ss+ (y_top<.8)+ (y_top > .5);
    ss(y_top < .3) = -1000;
    ss(~isConvex) = -1000;
    curScores(k) = max(ss);
end

poses = posemap(comps);
poses = poses(sel_);
faceSet = faceSet(sel_);
faceScores = imageSet.faceScores;
q = cat(1,imageData.train.faceLandmarks.dpmRect);
q = q(cur_t,6);
faceScores = faceScores(sel_);


% curScores((~isinf(poses)))= curScores((~isinf(poses)))+.001*abs(poses(~isinf(poses)));

rrr = (0*T_saliency.stds+T_saliency.means_inside-1.5*T_saliency.means_outside)';
curScores = curScores + .01*(rrr(sel_)');
% curScores = (rrr(sel_)')*1;

curScores(isnan(curScores)) = -1000;
curScores(faceScores<-.65) = -1000;
curScores(isnan(curScores)) = -1000;
curScores(~(~isinf(poses) & abs(poses)<=45)) = -1000;
% curScores = faceScores;
%curScores = curScores + lambda_pose*(~isinf(poses) & abs(poses)<=30)+lambda_faceScore*(faceScores>-.6);
% curScores = curScores+.01*faceScores;

showSorted(faceSet,curScores,20);
[prec,rec,aps] = calc_aps2(curScores',cur_t(sel_));
% 
% I = faces.train_faces{1};
% skinprob = computeSkinProbability(double(im2uint8(I)));
% normaliseskinprob = normalise(skinprob);

%%
[ss,iss] = sort(curScores,'descend');
% 
myImageSet = imageSet;
myImageSet.faceBoxes = imageSet.faceBoxes(sel_,:);
myImageSet.lipBoxes = imageSet.lipBoxes(sel_,:);
myImageSet.faceScores = imageSet.faceScores(sel_);
myImageSet.labels = imageSet.labels(sel_);
myImageSet.imageIDs = imageSet.imageIDs(sel_);
myImageSet.faceLandmarks = imageSet.faceLandmarks(sel_);

%%

%% new features to consider:
%% bounding box (area, depends on pose), orientation of intervening contours
% location, extent
% exact point of interface with mouth, is mouth occluded or not.
% does occlusion stop at mouth (below eye line), boundary ownership of
% occluder (if considering only a part of a curve, where does the remainder
% go? also, have a better face score. 
% find a u-shape (occluder), note that the contour doesn't have to be
% exactly convex if long enough, just intervening with mouth area
% depth of occlussion; there may be another occluder upper in the face
% prefer larger occluders
% occluder cannot be made mostly of expected face contours.
% need a better landmark localization face alignment scheme to assist above
% things.

% iss = find(cur_t(sel_));

for kk = 1:100
    kk
%     kk =15
    r = extractCandidateFeatures3(conf,myImageSet,sals(sel_),iss(kk),true);
    
    
    
    subplot(1,2,2); hold on;
%     r = cur_rs{iss(kk)};
%     if (isempty(r) || is
%         empty(r.bbox))
%         curScores = -inf;
%         continue;
%     end
    c_score = exp(-(r.bc(:,1)-.5).^2*10);
    horz_extent = r.horz_extent;
    ucmStrengths = r.ucmStrength;
    salStrengths = r.salStrength/255;
    isConvex = double(r.isConvex');
    bbox = r.bbox;
    y_top = bbox(:,2);
    ss = c_score+1*(horz_extent)+2*ucmStrengths+2*salStrengths;
%     ss = ss+ (y_top<.8)+ (y_top > .5);
    ss(y_top < .3) = -1000;
    ss(~isConvex) = -1000;
    curScores(k) = max(ss);
    [currentScores,iCurrentScores] = sort(ss,'descend');
    bb = r.bbox(iCurrentScores(1),:);
    bb([1 3]) = bb([1 3])*size(faceSet{iss(kk)},2);
    bb([2 4]) = bb([2 4])*size(faceSet{iss(kk)},1);
    plotBoxes2(bb([2 1 4 3]),'g--','LineWidth',2);
    pause
end

% showSorted(faces.train_faces(cur_t),faceScores);
   

% display some of the features...


%%



for q =809:length(imageSet.imageIDs)
    q
    k = iv(q);
    %     if (~cur_t(k))
    %         continue;
    %     end
    if (~cur_t(k))
        continue;
        ff = rand(1);
        if (ff > falseChance)
            continue;
        end
    end
    
    currentID = imageSet.imageIDs{k};
    % 1 : for cups / bottles
    isImg = cellfun(@any,strfind(imageNames,currentID));
    isObj = cellfun(@any,strfind(objNames,'cup'));
    isObj = isObj | cellfun(@any,strfind(objNames,'bottle'));
    f = find(isImg & isObj);
    % get the poly-mask...
    % filter candidates according to face pose...
    curComp = comps(k);
    a = posemap(curComp);
    
    if (isinf(a) || imageSet.faceScores(k) < -.95)
        continue
    end
    if (abs(a) >= 30)
        continue;
    end
    conf.get_full_image  = 0;
    % get the regions and corresponding UCM strengths
    %     [I_sub_color,face_mask,regions_sub,subUCM] = extractCandidateFeatures(conf,currentID,imageSet.faceBoxes(k,:),imageSet.lipBoxes(k,:),imageSet.faceLandmarks(k),debug_);
    [I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
    ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
    load(ucmFile); % ucm
    ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
    bbox = round(imageSet.faceBoxes(k,1:4));
    
    Rs = extractCandidateFeatures3(conf,imageSet,sal_train,k);
    
    subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
    I = I(bbox(2):bbox(4),bbox(1):bbox(3),:);
    imageSet.faceBoxes(k,3:4)-imageSet.faceBoxes(k,1:2);
    % sort ucm values, lineseg each one.
    u = unique(subUCM);
    
    clf;   subplot(1,2,1); imagesc(I); axis image; colormap('default');
    subplot(1,2,2); imagesc(subUCM); axis image;
    
    E = subUCM.*(subUCM>.1);
    [seglist,edgelist] = processEdges(E);
    
    candidates = findConvexArcs(seglist,E,edgelist);
    candidates = fixSegLists(candidates);
    % get the following features:
    % center, width, height, color, saliency
    
    bboxes = zeros(length(candidates),4);
    y_tops = zeros(length(candidates),1);
    ucmStrengths = zeros(length(candidates),1);
    %%%%%%%%
    
    
    
    %
    %     pFirst = cell2mat(cellfun(@(x) x(1,:),candidates,'UniformOutput',false)');
    %     pLast =  cell2mat(cellfun(@(x) x(end,:),candidates,'UniformOutput',false)');
    %     pMean = cell2mat(cellfun(@(x) mean(x),candidates,'UniformOutput',false)');
    %
    %     horzDist = (pLast(:,2)-pFirst(:,2))./size(I,2);
    %
    %     isConvex = false(size(candidates));
    %     u = pLast-pFirst;
    %
    %     % make sure that the base of the contour isn't too vertical
    %
    %     for ic = 1:length(candidates)
    %         v = bsxfun(@minus,candidates{ic}(2:end-1,:),pFirst(ic,:));
    %         curCross = u(ic,2)*v(:,1)-u(ic,1)*v(:,2);
    %         isConvex(ic) = ~any(curCross>0);
    %     end
    
    %     candidates = candidates(~isConvex);
    ims = cellfun2(@(x)(paintLines(zeros(size(E)),x)>0),candidates);
    % isConvex = true(size(isConvex)); %% TODO!!
    % calculate angle from top of image, check convexity
    topCenter = [size(E,2)/2,1];
    isConvex = false(size(candidates));
    for iCandidate = 1:length(candidates)
        %         iCandidate
        pts = candidates{iCandidate};
        pts = [pts(1:end,1:2);pts(end,3:4)];
        pts = fliplr(pts); % x,y
        d = bsxfun(@minus,pts,topCenter);
        tg = atan2(d(:,2),d(:,1));
        %         [x,y] = poly2cw(pts(:,1),pts(:,2));
        
        x = pts(:,1);
        y = pts(:,2);
        
        [s,is] = sort(tg,'ascend');
        %         clf,imagesc(E); hold on; plot(pts(:,1),pts(:,2),'r-s');
        %         quiver(topCenter(1),topCenter(2),x(is(1))-topCenter(1),y(is(1))-topCenter(2),0,'r');
        %         quiver(topCenter(1),topCenter(2),x(is(end))-topCenter(1),y(is(end))-topCenter(2),0,'g');
        
        if (is(1) > is(end))
            inc = -1;
        else
            inc = 1;
        end
        x = x(is(1):inc:is(end));
        y = y(is(1):inc:is(end));
        
        pts = [x y];
        diffs = diff(pts);
        crosses = zeros(size(diffs,1)-1,1);
        for id = 1:length(crosses)
            crosses(id) = diffs(id,2)*diffs(id+1,1)-diffs(id,1)*diffs(id+1,2);
        end
        
        isConvex(iCandidate) = ~any(crosses<0);
        
        
        %         clf,imagesc(E); hold on; plot(x,y,'r-s','LineWidth',2);
        %         plot(x(1),y(1),'g*');
        %         pause;
    end
    
%     mImage(ims);
%     mImage(ims(isConvex));
    %%%%%%%%
    ims = ims(isConvex);
    candidates = candidates(isConvex);
    
    for iCandidate = 1:length(candidates)
        lineImage = ims{iCandidate};
        [y,x] = find(lineImage);
        bboxes(iCandidate,:) = pts2Box([x y]);
        %         chull = convhull(x,y);
        ucmStrengths(iCandidate) = mean(subUCM(imdilate(lineImage,ones(3)) & subUCM > 0));
        % find the x,y which span the x extend, from the top. this is the
        % "top" of the cup.
        [y,is] = sort(y,'ascend');
        x = x(is);
        xmin = min(x);
        xmax = max(x);
        x_left = find(x==xmin,1,'first');
        x_right = find(x==xmax,1,'first');
        y_tops(iCandidate) = max(y(x_left),y(x_right));
    end
    bboxes(:,[1 3]) = bboxes(:,[1 3])/size(E,2);
    bboxes(:,[2 4]) = bboxes(:,[2 4])/size(E,1);
    bc = boxCenters(bboxes);    
    c_score = exp(-(bc(:,1)-.5).^2*10);
    horz_extent = bboxes(:,3)-bboxes(:,1);    
    curScore = c_score+horz_extent+ucmStrengths;
    %     showSorted(ims,curScore);
    y_tops = y_tops/size(E,1);
    %  pause
    
    
end

%%
ucmStrengths = cellfun(@(x) mean(subUCM(imdilate(x,ones(3)) & subUCM > 0)),regions_sub);
conf.get_full_image = 1;
% get the shape mask;
if (~cur_t(k))
    %         if (ff < falseChance)
    falseCount = falseCount + 1;
    if (a > 0) % flip.
        I_sub_color = flip_image(I_sub_color);
        I_sub_shape = flip_image(I_sub_shape);
        subUCM = flip_image(subUCM);
        falses(falseCount).metadata.flipped = true;
    else
        falses(falseCount).metadata.flipped = false;
    end
    falses(falseCount).img = I_sub_color;
    falses(falseCount).mask = regions_sub;
    falses(falseCount).ucm = subUCM;
    falses(falseCount).metadata.pose = a;
    falses(falseCount).metadata.faceScore = imageSet.faceScores(k);
    falses(falseCount).ucmStrengths = ucmStrengths;
    %         end
else
    [ I,xmin,xmax,ymin,ymax ] = getImage( conf,currentID);
    box_c = round(boxCenters(imageSet.lipBoxes(k,:)));
    sz = imageSet.faceBoxes(k,3:4)-imageSet.faceBoxes(k,1:2);
    bbox = round(inflatebbox([box_c box_c],floor(sz/1.5),'both',true));
    bw = zeros(ymax-ymin+1,xmax-xmin+1);
    for iF = 1:length(f)
        bw = bw | poly2mask(objectSamples(f(iF)).polygon.x-xmin,objectSamples(f(iF)).polygon.y-ymin,ymax-ymin+1,xmax-xmin+1);
    end
    %     I = I(ymin:ymax,xmin:xmax,:);
    I_sub_shape = cropper(bw,bbox);
    
    if (nnz(I_sub_shape)==0)
        continue;
    end
    count_ = count_+1;
    if (a > 0) % flip.
        I_sub_color = flip_image(I_sub_color);
        I_sub_shape = flip_image(I_sub_shape);
        subUCM = flip_image(subUCM);
        featureData(count_).metadata.flipped = true;
    else
        featureData(count_).metadata.flipped = false;
    end
    
    featureData(count_).img = I_sub_color;
    featureData(count_).mask = I_sub_shape;
    featureData(count_).ucm = subUCM;
    featureData(count_).metadata.pose = a;
    featureData(count_).metadata.faceScore = imageSet.faceScores(k);
    
    x = bwperim(I_sub_shape);
    ucmStrengths = mean(subUCM(imdilate(x,ones(3)) & subUCM > 0));
    featureData(count_).ucmStrengths = ucmStrengths;
end
%     clf;subplot(1,2,1);imagesc(I_sub_color); axis image
%     subplot(1,2,2);imagesc(I_sub_shape); axis image
%     pause;
%     continue;
%     shapeFeatures = sfe.extractFeatures(I_sub_color,{I_sub_shape});
% end

for k = 1:length(featureData)
    k
    if (~isempty(featureData(k).mask))
        featureData(k).shapeFeatures = sfe.extractFeatures(featureData(k).img,featureData(k).mask);
    end
end

for k = 1:length(falses)
    k
    if (~isempty(falses(k).mask))
        falses(k).shapeFeatures = sfe.extractFeatures(falses(k).img,falses(k).mask);
    end
end

%%
m = [featureData.metadata];
allPoses = [m.pose];
sel_ = abs(allPoses)>=45;

pos_feats = cat(2,featureData(sel_).shapeFeatures);
pos_feats = [pos_feats;[featureData(sel_).ucmStrengths]];
m_false = [falses.metadata];
sel_false = abs([m_false.pose]) >= 45;
neg_feats = cat(2,falses(sel_false).shapeFeatures);
neg_feats = [neg_feats;[falses(sel_false).ucmStrengths]];


[ii,jj] = find(isnan(neg_feats));
neg_feats(:,unique(jj(:))) = [];

[svm_model,ws,b] = train_classifier(pos_feats,neg_feats,.01,1,0);

%%
debug_ = false;
imageSet = imageData.test;
iv = 1:length(imageSet.imageIDs);
testFeats = {};
cur_t = imageSet.labels;
for q = 1:length(imageSet.imageIDs)
    q
    k = iv(q);
    if (cur_t(k))
        continue;
    end
    %
    
    currentID = imageSet.imageIDs{k};
    curComp = imageSet.faceLandmarks(k).c;
    if (isinf(curComp) || imageSet.faceScores(k) < -.95)
        continue
    end
    a = posemap(curComp);
    
    conf.get_full_image  = 0;
    % get the regions and corresponding UCM strengths
    [I_sub_color,face_mask,regions_sub,subUCM] = extractCandidateFeatures(conf,currentID,imageSet.faceBoxes(k,:),imageSet.lipBoxes(k,:),imageSet.faceLandmarks(k),debug_);
    
    if (a > 0) % flip.
        I_sub_color = flip_image(I_sub_color);
        I_sub_shape = flip_image(I_sub_shape);
        subUCM = flip_image(subUCM);
        %             falses(falseCount).metadata.flipped = true;
        
    end
    if (isempty(regions_sub))
        continue;
    end
    
    testFeats{k} = sfe.extractFeatures(I_sub_color,regions_sub);
    testFeats{k} = [testFeats{k};cellfun(@(x) mean(subUCM(imdilate(x,ones(3)) & subUCM > 0)),regions_sub)];
end


%%
scores = -1000*ones(size(cur_t));
ws(end) = .01;

for iFeat = 1:length(testFeats)
    iFeat
    curFeats = testFeats{iFeat};
    if (~isempty(curFeats))
        s = curFeats'*ws;
        s(isnan(s)) = -1000;
        scores(iFeat) = max(s);
        %     pause
    end
end

scores(scores==-1000) = min(scores(scores~=-1000));

comps_test = [imageSet.faceLandmarks.c];
comps_test(comps_test==0) = 14;
comps_test = posemap(comps_test);

scores = scores + (imageSet.faceScores > -.7)' + 0*(~isinf(comps_test) & abs(comps_test)<=15)';
scores(isnan(scores)) = -1000;
showSorted(faces.test_faces,scores,100);


%%

%%
save Rs_train Rs


imageSet = imageData.test;
iv = 1:length(imageSet.imageIDs);
debug_ = false;
Rs = {};
for q = 1:length(imageSet.imageIDs)
    q
    k = iv(q);
    if (~cur_t(k))
        continue;
    end
    currentID = imageSet.imageIDs{k};
    Rs{k} = extractCandidateFeatures(conf,currentID,imageSet.faceBoxes(k,:),imageSet.lipBoxes(k,:), imageSet.faceLandmarks(k),debug_);
end

save Rs_test Rs

%
%% 1. find all locations where the line is rather horizontal and spans a significant part of the image (cup1).

% 1 R = [contourLengths/size(E,1);...
%  2   ucmStrengths;...
%  3   row(pMean(:,1))/size(E,1);...
%  4   verticality';...
%  5   skinTransition;...
%  6   insides;...
%  7   nIntersections;...
%  8   areas./nnz(face_mask);...
%  9   inLips;...
%  10   horzDist';...
%  11  areas;...
%  12  bboxes'/size(E,1)];
%%
myScores = zeros(length(Rs),1);
%1  2   3   4   5   6   7   8   9   10  11  12  13  14  15
%weights = [1  5   0   0   20  1   0  0   10   10   0   0   0   0   0];

weights = [1  5   0   0   10  1   1  1   10   1   1  0   0   0   0];

opts = optimset;
opts.Display = 'iter';
opts.PlotFcns = @optimplotfval;

x = fminsearch(@(x) mySearchFun(Rs, cur_t, x),zeros(size(weights')),...
    opts);


cur_t = imageData.train.labels;
for k = 1:length(Rs)
    
    R = Rs{k};
    if (isempty(R))
        myScores(k) = -10;
        continue;
    end
    contourLengths = R(1,:);
    areas = R(8,:);
    inLips = R(9,:);
    ucmStrengths = R(2,:);
    t1 = contourLengths>.3;
    horzDist = R(10,:);
    %     t2 = areas>.1 & areas < .2;
    t2 = horzDist;
    %     curScores = t1+uecmStrengths+1*t2+inLips;        
    curScores = R'*weights';
    
    %     curScores(isnan(curScores)) = -100;
    
    myScores(k) = max(curScores);
end
myScores(isnan(myScores)) = -1000;
[v,iv] = sort(myScores,'descend');
[prec,rec,aps,T] = calc_aps2(myScores,cur_t);
% L1 = load('~/mircs/experiments/experiment_0001_improved/exp_result.mat');
%%
%
theScores = (L1.train_saliency.stds+L1.train_saliency.means_inside-L1.train_saliency.means_outside)';
theScores(isnan(theScores)) = -10000;
theScores = theScores+1*(imageSet.faceScores'>-.65)+.001*rand(size(theScores));
%(imageSet.faceScores > -.65)';

[prec,rec,aps,T] = calc_aps2(theScores,imageSet.labels);
%%
a1 = showSorted(faces.train_faces(imageSet.labels),theScores(imageSet.labels),100);
a2 = showSorted(sal_train(imageSet.labels),theScores(imageSet.labels),100);

a1 = imresize(a1,dsize(a2,1:2),'bilinear');

imagesc(a1); axis image;
figure,imagesc(a2); axis image;



%%

% X_train = imageSetFeatures2(conf,faces.train_faces,true,[64 64]);
% X_test = imageSetFeatures2(conf,faces.test_faces,true,[64 64]);
% load w.mat;



face_comp = [imageData.test.faceLandmarks.c]';
ww = X_test'*w;
% theScores = 0*myScores;
theScores = (myScores > 0);
theScores = theScores+.1*ww;
theScores = theScores + (L1.test_saliency.stds+L1.test_saliency.means_inside-L1.test_saliency.means_outside)';
theScores = theScores+0*ismember(face_comp,6:11)+1*(imageData.test.faceScores > -.7)';
theScores(isnan(theScores)) = -10000;
[prec,rec,aps,T] = calc_aps2(theScores,cur_t);
aps
% showSorted(faces.test_faces,theScores,100);


% showSorted(faces.test_faces,imageData.test.faceScores>-.65,1000);

%%
% bbb = imageSet.faceBoxes;
% sz1 = (bbb(:,3)-bbb(:,1));
%
% theScores = theScores+1000*(sz1>12);

M = showSorted(faces.train_faces,theScores,100);
% M = imresize(M,.5,'bicubic');
imwrite(M,'~/mircs/experiments/experiment_0005/drinking_new.png');