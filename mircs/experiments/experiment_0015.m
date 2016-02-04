%%%%%% Experiment 15 %%%%%%%
% Nov . 21, 2013

% learn the appearance of the occluding objects.

% function experiment_0015
if (~exist('toStart','var'))
    toStart = 1;
    initpath;
    config;
    resultDir = ['~/mircs/experiments/experiment_0015'];
    ensuredir(resultDir);
    conf.features.vlfeat.cellsize = 8;
    
    addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
    addpath('/home/amirro/code/3rdparty/sliding_segments');
    load ~/storage/misc/imageData_new; % which image-data to load? the one by zhu, or my face detection + mouth detection?
        
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
    facesPath = fullfile('~/mircs/experiments/common/faces_cropped_new.mat');
    load(facesPath);
    [objectSamples,objectNames] = getGroundTruth(conf,train_ids,train_labels);
    face_thresh = -.58;
end
%%
% sel_ = faceScores_t >= -.58;
% ids_t =ids_t(sel_);
% faceScores_t = faceScores_t(sel_);

% now for each image find the region of interest.

% stages:
% 0 : pre-compute descriptors densly around face areas X 2 for all dataset
% images.
% 1. for each object type :
% 1.1 find images containing object type
% 1.2 if image has face with score above thresold, (i.e, in ids_t),
% --> check if object near enough mouth area.
% compute features for object and add as positive example.
% add geometric features too such as bounding box relative to face,
% diameter, occupancy mask, etc.
% find regions from non-class images, (regions around face) and add
% as negative samples for training.
% train.


trainDataFile = '~/mircs/experiments/experiment_0015/regionData_train.mat';
if (exist(trainDataFile,'file'))
    load(trainDataFile);
else
    
    imageSet = imageData.train;
    objectClasses = {'cup','straw','bottle'};
    gt_objects = {objectSamples.name};
    newGT = struct('sourceImage',{},'polygon',{},'name',{});
    objCount = 0;
    
    for iObjType = 1:length(objectClasses)
        %
        objIndices = find(strncmp2(objectClasses{iObjType},gt_objects));
        
        for iObj = 1:length(objIndices)
            rec = objectSamples(objIndices(iObj));
            imageID = rec.sourceImage;
            imageIndex = find(strncmp2(imageID,imageSet.imageIDs));
            faceScore = imageSet.faceScores(imageIndex);
            % find if the image had a high enough face score.
            if (faceScore < face_thresh)
                disp('skipping : low face score');
                continue;
            end
            [I,rect] = getImage(conf,imageID);
            faceBox = imageSet.faceBoxes(imageIndex,:);
            x = rec.polygon.x;
            y = rec.polygon.y;
            gt_mask = poly2mask(x-rect(1),y-rect(2),size(I,1),size(I,2));
            ovp = boxRegionOverlap(faceBox,gt_mask);
            
            if (ovp == 0)
                disp('skipping : no overlap with face');
                continue;
            end
            
            %         face_pts = box2Pts(faceBox);
            %         face_mask = poly2mask(face_pts(:,1),face_pts(:,2),size(I,1),size(I,2));
            roi = gt_mask;
            objCount = objCount+1;
            newGT(objCount).name = objectClasses{iObjType};
            perim = bwboundaries(roi);
            perim = fliplr(perim{1});
            newGT(objCount).sourceImage = imageID;
            newGT(objCount).polygon.x = perim(:,1);
            newGT(objCount).polygon.y = perim(:,2);
            newGT(objCount).partID = rec.partID;
            %         pause;
        end
    end
    
    % discover negative regions: regions (or just including regions) within
    % negative faces image), then extract features
    
    imageSet = imageData.train;
    
    regionData_train = struct('regions',{},'regionGroups',{},'ref_box',{},'face_mask',{},'mouth_mask',{},'class',{},...
        'imageIndex',{});
    true_images = {newGT.sourceImage};
    t_count = 0;
    
    for k = 1:length(imageSet.imageIDs)
        k
        currentID = imageSet.imageIDs{k};
        [regions,groups,ref_box,face_mask,mouth_mask,I,params] = extractRegions(conf,imageSet,k);
        [curImage,imageRect] = getImage(conf,currentID);
        % if a positive image, extract only the relevant region
        part_id = 0;
        f = find(strncmp2(currentID,true_images));
        if (~isempty(f) && imageSet.labels(k))
            for iF = 1:length(f)
                t_count = t_count + 1;
                rec = newGT(f(iF));
                bw = poly2mask2([rec.polygon.x,rec.polygon.y],size(curImage));
                regions = {cropper(bw,ref_box)};
                regionData_train(t_count).regions = regions;
                regionData_train(t_count).ref_box = ref_box;
                regionData_train(t_count).face_mask = face_mask;
                regionData_train(t_count).mouth_mask = mouth_mask;
                regionData_train(t_count).class = rec.partID;
                regionData_train(t_count).regionGroups = [];
                regionData_train(t_count).imageIndex = k;
            end
        else
            t_count = t_count + 1;
            regionData_train(t_count).regions = regions;
            regionData_train(t_count).ref_box = ref_box;
            regionData_train(t_count).face_mask = face_mask;
            regionData_train(t_count).mouth_mask = mouth_mask;
            regionData_train(t_count).class = 0;
            regionData_train(t_count).regionGroups = groups;
            regionData_train(t_count).imageIndex = k;
        end
    end
    
    save ~/mircs/experiments/experiment_0015/regionData_train.mat regionData_train
end

testDataFile = '~/mircs/experiments/experiment_0015/regionData_test.mat';
if (exist(testDataFile,'file'))
    load(testDataFile);
else
    % do the same for test...
    regionData_test = struct('regions',{},'regionGroups',{},'ref_box',{},'face_mask',{},'mouth_mask',{},'class',{},...
        'imageIndex',{});
    imageSet = imageData.test;
    
    t_count = 0;
    
    for k = 1:length(imageSet.imageIDs)
        k
        currentID = imageSet.imageIDs{k};
        [regions,groups,ref_box,face_mask,mouth_mask,I,params] = extractRegions(conf,imageSet,k);
        [curImage,imageRect] = getImage(conf,currentID);
        % if a positive image, extract only the relevant region
        part_id = 0;
        t_count = t_count + 1;
        regionData_test(t_count).regions = regions;
        regionData_test(t_count).ref_box = ref_box;
        regionData_test(t_count).face_mask = face_mask;
        regionData_test(t_count).mouth_mask = mouth_mask;
        regionData_test(t_count).class = imageSet.labels(k);
        regionData_test(t_count).regionGroups = groups;
        regionData_test(t_count).imageIndex = k;
    end
    % just a small fix, no longer relevant
%     for k = 1:length(regionData_test)
%         regionData_test(k).class = imageSet.labels(regionData_test(k).imageIndex);
%     end

    save ~/mircs/experiments/experiment_0015/regionData_test.mat regionData_test
end

train_feats_file = '~/mircs/experiments/experiment_0015/feats_train.mat';
if (~exist(train_feats_file,'file'))
    indRange = [];
    toDebug = false;
    minFaceScore = -.7;
    [feats_train,labels_train] = extractRegionFeatures(conf,regionData_train,imageData.train,toDebug,indRange,minFaceScore);
    save ~/mircs/experiments/experiment_0015/feats_train.mat feats_train 
else
    load(train_feats_file);
end

load ~/mircs/experiments/experiment_0015/regionData_train.mat
load ~/mircs/experiments/experiment_0015/regionData_test.mat

% just debugging
[feats_train,labels_train] = extractRegionFeatures(conf,regionData_train,imageData.train,true);
[feats_test,labels_test] = extractRegionFeatures(conf,regionData_test,imageData.test,true);

load /home/amirro/mircs/experiments/experiment_0015/feats_train2.mat
load /home/amirro/mircs/experiments/experiment_0015/feats_test2.mat

[X_train,labels_train] = feats_to_feat_vectors(feats_train);

[X_test,labels_test] = feats_to_feat_vectors(feats_test);
X_train = X_train';
X_test = X_test';

% save /home/amirro/mircs/experiments/experiment_0015/X.mat X_train X_test labels_train labels_test

f_pos = find(labels_train==1); % labels_train==3);
% f_pos = find(labels_train);
f_neg = find(~labels_train);
f_neg = vl_colsubset(f_neg,5000);


pBoost = struct('verbose',1,'nWeak',128,'pTree',struct('maxDepth',2));
pBoost.discrete = 0;
model = adaBoostTrain(X_train(f_neg,:),X_train(f_pos,:),pBoost);

hs = adaBoostApply(X_test,model);

testSet = imageData.test;
testIDS = testSet.imageIDs;
imageScores = -1000*ones(1,length(testIDS));
for k = 1:length(feats_test)
    imageIndex=  feats_test(k).imageIndex;
    imageScores(imageIndex) = max(imageScores(imageIndex),hs(k));
end

showSorted(faces.test_faces,imageScores,50)

[prec,rec,aps] = calc_aps2(imageScores',imageData.test.labels);

train_faces = cellfun2(@(x) imResample(x,[64 64],'bilinear'),faces.train_faces);
pChns = chnsCompute;
pChns.pColor.enabled = 0;
chnData = cellfun2(@(x) chnsCompute(x,pChns),train_faces);
chnData = cellfun2(@(x) row(cat(3,x.data{:})),chnData);
chnData = cat(1,chnData{:});
dpmRects = cat(1,imageData.train.faceLandmarks.dpmRect);
dpmScores = -100*ones(1,length(imageData.train.imageIDs));
for k = 1:length(dpmScores)
    dd = imageData.train.faceLandmarks(k).dpmRect;
    if (~isempty(dd))
        dpmScores(k) = dd(6);
    end
end
dpmScores(dpmScores==-100) = min(dpmScores(dpmScores>-100));

chnData1 = [chnData,imageData.train.faceScores',[imageData.train.faceLandmarks.c]',dpmScores'];
    
pBoost = struct('verbose',1,'nWeak',128,'pTree',struct('maxDepth',2));
model_chn = adaBoostTrain(chnData1(~imageData.train.labels,:),chnData1(imageData.train.labels,:),pBoost);
% plot(model.losses)

test_faces = cellfun2(@(x) imResample(x,[64 64],'bilinear'),faces.test_faces);
chnData_test = cellfun2(@(x) chnsCompute(x,pChns),test_faces);
chnData_test = cellfun2(@(x) row(cat(3,x.data{:})),chnData_test);
chnData_test = cat(1,chnData_test{:});


dpmScores_test = -100*ones(1,length(imageData.test.imageIDs));
for k = 1:length(dpmScores)
    dd = imageData.test.faceLandmarks(k).dpmRect;
    if (~isempty(dd))
        dpmScores_test(k) = dd(6);
    end
end
dpmScores_test(dpmScores_test==-100) = min(dpmScores_test(dpmScores_test>-100));

chnData_test1 = [chnData_test,imageData.test.faceScores',[imageData.test.faceLandmarks.c]',dpmScores_test'];

hs = adaBoostApply(chnData_test1,model_chn);

figure,plot(hs)
showSorted(test_faces,hs,150)
[prec,rec,aps] = calc_aps2(hs,imageData.test.labels);
plot(rec,prec)

totalData_train = zeros(size(X_train,1),size(chnData1,2)+size(X_train,2));
totalData_train(:,1:size(X_train,2)) = X_train;
totalData_train(:,size(X_train,2)+1:end) = chnData1([feats_train.imageIndex],:);

pBoost = struct('verbose',1,'nWeak',128,'pTree',struct('maxDepth',2));
%X_train(f_neg,:),X_train(f_pos,:)
model_full = adaBoostTrain(totalData_train(f_neg,:),totalData_train(f_pos,:),pBoost);
% plot(model.losses)

totalData_test = zeros(size(X_test,1),size(chnData_test1,2)+size(X_test,2));
totalData_test(:,1:size(X_test,2)) = X_test;
totalData_test(:,size(X_test,2)+1:end) = chnData_test1([feats_test.imageIndex],:);

hs = adaBoostApply(totalData_test,model_full);

imageScores = -1000*ones(1,length(testIDS));
for k = 1:length(feats_test)
    imageIndex=  feats_test(k).imageIndex;
    imageScores(imageIndex) = max(imageScores(imageIndex),hs(k));
end
showSorted(faces.test_faces,imageScores,50)
1
imageScores(imageScores==-1000) = min(imageScores(imageScores>-1000));

[prec,rec,aps] = calc_aps2(imageScores',imageData.test.labels);


%% ok, as taking all features didn't bring up great results, I'll try to add some
%% filters of my own, for example regarding the face score and component.

feat_name = 'train';

load(sprintf('/home/amirro/mircs/experiments/experiment_0015/feats_%s2.mat',feat_name));
% for k = 1:length(feats_train)
%     feats_train(k).region_occupancy = feats_train(k).region_occupancy>0;
% end
%%
[X_train,labels_train] = feats_to_feat_vectors(feats_train);
imageSet = imageData.train;
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTrain.mat');
train_mouth_scores = L_det.newDets.cluster_locs([feats_train.imageIndex],12);
[F_train,classes_train,ticklabels] = feats_to_featureset2(feats_train,L_det);

X_train = [X_train;train_mouth_scores';F_train];

w = [1 1 1 1 -5 5 1 1 .5 5 -1 -5];
h_train = w*F_train;

%X_train(f_neg,:),X_train(f_pos,:)
%Q_train = [F_train;X_train([1:86,88:end],:)];

% h_train = w*F_train;

% Q_train = [F_train;X_train;h_train];
%%
pBoost = struct('verbose',1,'nWeak',512,'pTree',struct('maxDepth',2));

% Q_train = F_train;
%sel1 = ismember(classes_train,[1 3]);
sel1 = classes_train > 0;

% Q_pos = Q_train(:,sel1)';
% Q_pos = repmat(Q_pos,100,1);


model_full = adaBoostTrain(Q_train(:,classes_train==0)',Q_pos,pBoost);

% plot(model_full.losses)
% imagesc(model_full.fids); colorbar
% imagesc(model_full.fids==3);
% imagesc(model_full.depth)


% Q_pos = ;


% Q_pos = Q_train(:,sel1 & h_train >= 4);
% 
classifier = train_classifier_pegasos(X_train(:,sel1),X_train(:,~sel1),-1);

% train a classifier on the top scoring images by the heuristic.

% plot(sort(h_train))
%%
feat_name = 'test';
load(sprintf('/home/amirro/mircs/experiments/experiment_0015/feats_%s2.mat',feat_name));
load ~/mircs/experiments/experiment_0015/regionData_test.mat
% for k = 1:length(feats_test)
%     feats_test(k).region_occupancy = feats_test(k).region_occupancy>0;
% end
%%
[X_test,labels_test] = feats_to_feat_vectors(feats_test);
% 
% 
imageSet = imageData.test;
imageIDS = imageSet.imageIDs;
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTest.mat');
[F_test,classes_test,ticklabels] = feats_to_featureset2(feats_test,L_det);
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTest.mat');
TTT = L_det.newDets.cluster_locs(:,12);
test_mouth_scores = L_det.newDets.cluster_locs([feats_test.imageIndex],12);
% % % X_test = [X_test;test_mouth_scores';F_test];
% h_test = w*F_test;%5*classifier.w(1:end-1)'*Q_test;
% Q_test = [F_test;X_test;h_test];

%%
Q_pos = F_train(classes_train > 0)
classifier = train_classifier_pegasos(F_train(:,classes_train>0),F_train(:,classes_train==0),-1);
%%
feats = feats_test;
hs = w*F_test;%5*classifier.w(1:end-1)'*Q_test;
% hs = classifier.w(1:end-1)'*F_test;
% hs = classifier.w(1:end-1)'*X_test;
imageScores = -inf*ones(1,length(imageIDS));
bestImageRegion = zeros(size(imageScores));
for k = 1:length(feats)
    imageIndex = feats(k).imageIndex;
    [m,im] = max([imageScores(imageIndex),hs(k)],[],2);
    imageScores(imageIndex) = m;
    if (im == 2)
        bestImageRegion(imageIndex) = k;
    end   
    %imageScores(imageIndex) = max(imageScores(imageIndex),hs(k));
end
% show the region features for each of the top regions.
imageScores = imageScores -6*TTT';
imageScores = clip_to_min(imageScores);

T_1 = -.4;
imageScores(imageData.test.faceScores <= T_1) = imageScores(imageData.test.faceScores <= T_1)-1;

%% for train
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTrain.mat');
TTT = L_det.newDets.cluster_locs(:,12);

feats = feats_train;
hs = w*F_train;%5*classifier.w(1:end-1)'*Q_test;
% hs = classifier.w(1:end-1)'*F_test;
% hs = classifier.w(1:end-1)'*X_test;
imageScores = -inf*ones(1,length(imageData.train.imageIDs));
bestImageRegion = zeros(size(imageScores));
for k = 1:length(feats)
    imageIndex = feats(k).imageIndex;
    [m,im] = max([imageScores(imageIndex),hs(k)],[],2);
    imageScores(imageIndex) = m;
    if (im == 2)
        bestImageRegion(imageIndex) = k;
    end   
    %imageScores(imageIndex) = max(imageScores(imageIndex),hs(k));
end
% show the region features for each of the top regions.
imageScores = imageScores -6*TTT';
imageScores = clip_to_min(imageScores);

T_1 = -.4;
imageScores(imageData.train.faceScores <= T_1) = imageScores(imageData.train.faceScores <= T_1)-1;


%plot(imageSco
%plot(imageScores>=4)
% qq = zeros(size(imageScores));
% for k = 1:length(bestImageRegion)
%     if (bestImageRegion(k) > 0)
%         qq(k) = classifier.w(1:end-1)'*Q_test(:,bestImageRegion(k));
%     end 
% end
% imageScores(imageScores>=4) = imageScores(imageScores>=4) + 100+qq(imageScores>=4);
% qq = classifier.w(1:end-1)'*Q_test(:,bestImageRegion);

[r,ir] = sort(imageScores,'descend');
q = ir(1:150);
plot(cumsum(imageData.train.labels(ir)))
qq = bestImageRegion(q);
[prec,rec,aps] = calc_aps2(imageScores',imageData.test.labels);
save occ_scores_train.mat imageScores;
imageScores_mine = imageScores;
%plot(cumsum(imageData.test.labels(ir)))

%%
% add entire image bow.
conf.get_full_image = true;
learnParams = getDefaultLearningParams(conf);
conf.get_full_image = false;
featureExtractor = learnParams.featureExtractors{1};

bow_train = getEntireImageBOW(conf,train_ids,featureExtractor);
bow_test = getEntireImageBOW(conf,test_ids,featureExtractor);

save ~/mircs/experiments/experiment_0015/bow_entire_images.mat bow_train bow_test;

classifier = train_classifier_pegasos(repmat(bow_train(:,imageData.train.labels),1,20),bow_train(:,~imageData.train.labels));

T1 =classifier.w(1:end-1)'*bow_test;
[prec,rec,aps] = calc_aps2(T1',imageData.test.labels);
%%
face_occs = cat(2,feats.face_occupancy); % 5
region_occ = cat(2,feats.region_occupancy);

bestFaces = reshape(face_occs(:,qq),5,5,[]);
figure(2); montage3(bestFaces);title('face occs');colormap gray; axis image;

set(gcf,'name','face occs');

bestRegions = reshape(region_occ(:,qq),5,5,[]);
figure(3); montage3(bestRegions);title('face occs');colormap gray; axis image;

set(gcf,'name','region occs');

figure(4); montage3(faces.test_faces(q));

figure(5); imagesc(F_test(:,qq));

imagesc([F_test(:,qq);sum(F_test(:,qq))])
% ylim ytick yticklabel
set(gca,'YTickLabel',ticklabels);
%%
%Q_test = [F_test;X_test([1:86,88:end],:)];

% conf,regionData,imageSet,toDebug_,...
%     indRange,minFaceScore,saveRegions
%%
% extract the actual regions. 
%load ~/mircs/experiments/experiment_0015/regionData_test.mat
for k = 1:100
    k
    iq = qq(k);
    regionToGet = feats_test(iq).regionID;
    imageIndex = feats_test(iq).imageIndex;
    R = extractRegionFeatures(conf,regionData_test,imageData.test,false,imageIndex,-.1,regionToGet);
    if (~isfield(R,'region'))
        continue;
    end
%     displayRegions(im2double(faces.test_faces{q(k)}),{R.region});
    figure(1);clf;subplot(1,2,1);imagesc(R.region);axis image;
    subplot(1,2,2);imagesc(faces.test_faces{q(k)});axis image;
    
    % show all the points on the face as well....
    hold on;
    xy = imageData.test.faceLandmarks(imageIndex).xy;
%     if (isempty(xy))
%         return;
%     end
    xy_c = boxCenters(xy);            
    if (size(xy_c,1)==68)
        outline_ = [68:-1:61 52:60 16:20 31:-1:27];
        inner_lips = [36 37 38 42 42 45 47 49 36];
        outer_lips = [35 34 33 32 39 40 41 44 46 51 48 50 35];
        mouth_corner = [35 41];
    else
        outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
        inner_lips = [25 24 26 23 27];
        outer_lips = [16:22];
        mouth_corner = 19;
    end
    
    xy_c = bsxfun(@minus,xy_c,imageData.test.faceBoxes(imageIndex,1:2));
    
    plot(xy_c(outline_,1),xy_c(outline_,2),'m-','LineWidth',2);
    plot(xy_c(outer_lips,1),xy_c(outer_lips,2),'g--','LineWidth',2);
    plot(xy_c(:,1),xy_c(:,2),'rd'); 
    
    
    xy_c = boxCenters(xy);   
     xy_c = bsxfun(@minus,xy_c,regionData_test(imageIndex).ref_box(1:2));
    subplot(1,2,1);hold on;
    plot(xy_c(outline_,1),xy_c(outline_,2),'m-','LineWidth',2);
    plot(xy_c(outer_lips,1),xy_c(outer_lips,2),'g--','LineWidth',2);
    plot(xy_c(:,1),xy_c(:,2),'rd');
    
    
    %16/12/2013 now load the ellipse detection results....
    %     chull = 1:size(xy_c);
    % find the occluder!! :-)
%     
%     c_boxes = xy(outline_,:);
%     c_boxes = c_boxes-repmat(bbox(1:2),size(c_boxes,1),2);
%     c_poly = boxCenters(c_boxes);
% % % % %     saveas(gcf,fullfile(resultDir,'occlusion_samples',sprintf('%03.0f.png',k)));
%     pause;
    
end
%%
for k = 29:100
    k
    iq = qq(k);
   
    imageIndex = feats_test(iq).imageIndex;
    figure(2);clf;imagesc(faces.test_faces{q(k)}); axis equal;
    
    % show all the points on the face as well....
    hold on;
    xy = imageData.test.faceLandmarks(imageIndex).xy;
%     if (isempty(xy))
%         return;
%     end
    xy_c = boxCenters(xy);            
    if (size(xy_c,1)==68)
        outline_ = [68:-1:61 52:60 16:20 31:-1:27];
        inner_lips = [36 37 38 42 42 45 47 49 36];
        outer_lips = [35 34 33 32 39 40 41 44 46 51 48 50 35];
        mouth_corner = [35 41];
    else
        outline_ = [6:-1:1 16 25 27 22 28:39 15:-1:12];
        inner_lips = [25 24 26 23 27];
        outer_lips = [16:22];
        mouth_corner = 19;
    end
    
    xy_c = bsxfun(@minus,xy_c,imageData.test.faceBoxes(imageIndex,1:2));
    
    plot(xy_c(outline_,1),xy_c(outline_,2),'m-','LineWidth',2);
    plot(xy_c(outer_lips,1),xy_c(outer_lips,2),'g--','LineWidth',2);
    plot(xy_c(:,1),xy_c(:,2),'rd'); 
    
%     
%     xy_c = boxCenters(xy);   
%      xy_c = bsxfun(@minus,xy_c,regionData_test(imageIndex).ref_box(1:2));
%     figure(1); hold on;
%     plot(xy_c(outline_,1),xy_c(outline_,2),'m-','LineWidth',2);
%     plot(xy_c(outer_lips,1),xy_c(outer_lips,2),'g--','LineWidth',2);
%     plot(xy_c(:,1),xy_c(:,2),'rd');
%     %     chull = 1:size(xy_c);
%     % find the occluder!! :-)
% %     
%     c_boxes = xy(outline_,:);
%     c_boxes = c_boxes-repmat(bbox(1:2),size(c_boxes,1),2);
%     c_poly = boxCenters(c_boxes);
    
    pause;
    
end
%%
% Q_test = F_test;
hs = adaBoostApply(Q_test',model_full);


imageScores = -1000*ones(1,length(imageData.test.imageIDs));
bestImageRegion = zeros(size(imageScores));
for k = 1:length(feats_test)
    imageIndex=  feats_test(k).imageIndex;
    [m,im] = max([imageScores(imageIndex),hs(k)],[],2);
    imageScores(imageIndex) = m;
    if (im == 2)
        bestImageRegion(imageIndex) = k;
    end   
end


% find the index of each region in the corresponding image. 
perImageInds  = zeros(size(hs));
maxImageIndex = zeros(size(imageScores));
for k = 1:length(feats_test)
    feats_test(k).imageIndex
end
    


imageScores1 = clip_to_min(imageScores);


%%
imageScores = 0*(imageScores_mine)+ (imageScores1)-50*(TTT)'+10*T1;
TT = -.7;
imageScores(imageData.test.faceScores < TT) = imageScores(imageData.test.faceScores < TT)-50;
% imageScores = imageScores1;
showSorted(faces.test_faces,imageScores,150)

%imageScores(imageScores==-1000) = min(imageScores(imageScores>-1000));

[prec,rec,aps] = calc_aps2(imageScores',imageData.test.labels);
[r,ir] = sort(imageScores,'descend');
q = ir(1:150);
% q = q(71)
qq = bestImageRegion(q);

%%
%qq = qq(71)
face_occs = cat(2,feats_test.face_occupancy); % 5
region_occ = cat(2,feats_test.region_occupancy);

bestFaces = reshape(face_occs(:,qq),5,5,[]);
% figure(2); montage3(bestFaces);title('face occs');colormap gray; axis image;
set(gcf,'name','face occs');

bestRegions = reshape(region_occ(:,qq),5,5,[]);
figure(3); montage3(bestRegions);title('face occs');colormap gray; axis image;
set(gcf,'name','region occs');
% figure,plot(X_test(87,qq))
figure(4); montage3(faces.test_faces(q));

%% 
load ~/mircs/experiments/experiment_0012/lipImagesTrain.mat;

montage3(lipImages_orig(imageData.train.labels));

montage3(lipImages_orig(imageData.train.labels & imageData.train.faceScores' >-.7...
& abs([imageData.train.faceLandmarks.c]-7)'<=3))

montage3(lipImages_orig(imageData.train.labels & imageData.train.faceScores' >-.7...
& abs([imageData.train.faceLandmarks.c]-7)'>3))



