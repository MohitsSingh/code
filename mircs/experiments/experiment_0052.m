
%% 7/12/2014
% tell apart action classes by close inspection of the relevant segments.
% now we have for each image it's
% 1. location of face
% 2. facial landmarks
% 3. segmentation
% 4. saliency
% 5. location of action object (pixel-wise mask)
% 6. prediction of location of action object, learned separately.
if (~exist('initialized','var'))
    initpath;
    config;
    conf.get_full_image = true;
%     [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    initialized = true;
    conf.get_full_image = true;
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('~/storage/mircs_18_11_2014/allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);
    
    
    s40_fra_orig = s40_fra_faces_d;
    fra_db = s40_fra;
    net = init_nn_network();
    nImages = length(s40_fra);
    top_face_scores = zeros(nImages,1);
    for t = 1:nImages
        top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
    end
    min_face_score = 0;
    img_sel_score = col(top_face_scores > min_face_score);
    fra_db = s40_fra;
    top_face_scores_sel = top_face_scores(img_sel_score);
        
    % inialize parameters for various modules:
    % facial landmark parameters
    landmarkParams = load('~/storage/misc/kp_pred_data.mat');
    landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX);
    landmarkParams.conf = conf;
    landmarkParams.wSize = 96;
    landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
    landmarkParams.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    
    % segmentation...
    addpath '/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained';install;
    nn_net = init_nn_network();
    clear predData;
    
    %%%%% object prediction data
    %     load ~/code/mircs/s40_fra.mat;
    objPredData = load('~/storage/misc/actionObjectPredData.mat');
    objPredData.kdtree = vl_kdtreebuild(objPredData.XX,'Distance','L2');
    %
    
    %%%%% edge boxes demo
    addpath('/home/amirro/code/3rdparty/edgeBoxes/');
    model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    
    %%%%% set up opts for edgeBoxes (see edgeBoxes.m)
    opts = edgeBoxes;
    opts.alpha = .65; % step size of sliding window search
    opts.beta  = .75;     % nms threshold for object proposals
    opts.minScore = .01;  % min score of boxes to detect
    opts.maxBoxes = 1e4;  % max number of boxes to detect                
end
%%
% initialize parameters for different feature types.
params = defaultPipelineParams();
%
train_set = find([fra_db.isTrain]);
train_set = train_set(1:2:end);
test_set = find(~[fra_db.isTrain]);
val_set = setdiff(find([fra_db.isTrain]),train_set);

%
% aggregate features and labels, from valid images only, for now
% valids = {};
% u = {};
% validImages = cellfun(@(x) x.valid,all_train_feats);
% train a set of classifiers
clear stage_params
default_params = defaultPipelineParams(true);
stage_params(1) = defaultPipelineParams(false);
stage_params(1).learning.classifierType = 'svm';
stage_params(1).learning.classifierParams.useKerMap = false;
dataDir = '~/storage/s40_fra_feature_pipeline_stage_1';
stage_params(1).dataDir = dataDir;
% [labels,features,ovps] = collectFeatures(conf,all_train_feats,stage_params(1).params.features);
classes = [conf.class_enum.DRINKING conf.class_enum.SMOKING conf.class_enum.BLOWING_BUBBLES conf.class_enum.BRUSHING_TEETH];
top_k_false = 100;
sets = {train_set,val_set};
% set_results = {};
classifiers = {};

curParams = stage_params(1);
ind_in_orig = {};
debugging = false;
clear all_results;

debugging = true;

curParams.landmarkParams = landmarkParams;
curParams.features.nn_net = nn_net;
curParams.objPredData = objPredData;
curParams.featsDir = '~/storage/s40/feature_pipeline';

%%
% k = findImageIndex(fra_db,'drinking_001.jpg');

outDir = '~/s40/feature_pipeline/';

for t = 1:length(fra_db)
    [regionFeats,imageFeats,selected_regions] = extract_all_features(conf,fra_db(t),curParams);
end

stage_1_classifier_data = train_stage_classifier(conf,fra_db([fra_db.isTrain]),curParams);


k = 5142;
imgData = fra_db(k);

[regionFeats,imageFeats,selected_regions] = extract_all_features(conf,imgData,curParams);
I = imageFeats.I;
% [regions,ovp,sel_] = chooseRegion(I,selected_regions,.5)
% displayRegions(I,regions,ovp);

x2(I);
% x2(imageFeats.predictions.objPredictionImage)
plotBoxes(imageFeats.kp_preds);
displayRegions(I,selected_regions,[],.1);
%%
sel_train = [fra_db.isTrain] & img_sel_score' & [fra_db.classID] == conf.class_enum.DRINKING;
sel_train = sel_train | [fra_db.classID] == conf.class_enum.SMOKING;
sel_train = sel_train | [fra_db.classID] == conf.class_enum.BLOWING_BUBBLES;
sel_train = sel_train | [fra_db.classID] == conf.class_enum.BRUSHING_TEETH;
curParams.learning.nNegsPerPos = inf;
fra_db_train = fra_db(sel_train);
% fra_db_train = fra_db_train(1:10:end);
curParams.featsDir = '~/storage/s40/feature_pipeline';
[train_feats,train_labels,train_ovps,train_img_inds] = collect_feature_subset(conf,fra_db_train,curParams,inf);
train_feats = cat(2,train_feats{:});
train_labels = cat(2,train_labels{:});
train_ovps = cat(2,train_ovps{:});
train_img_inds = cat(2,train_img_inds{:});
u = unique(train_img_inds);

t1 = u(floor(end/2));
t2 = t1+1;
t1_ = train_img_inds<=t1;
t2_ = train_img_inds>=t2;
train_feats1 = train_feats;
% train_feats1 = vl_homkermap(train_feats1,1);
%[w b] = vl_svmtrain(train_feats1,train_labels(:,t1_)',.0000001);

%[w b] = vl_svmtrain(train_feats,2*(train_labels'~=-1)-1,.001);
train_labels_1 = 2*(train_ovps>=.5)-1;
[train_feats1,train_labels_1] = balanceData(train_feats1,train_labels_1,-1);
% classifier = train_classifier_pegasos(posFeats,negFeats,0);

% [w b] = vl_svmtrain(train_feats1,train_labels_1,.00001);

pBoost = struct('nWeak',round(128),'verbose',1,'pTree',struct('maxDepth',2,'nThreads',8),'discrete',1);
classifier = adaBoostTrain(train_feats1(:,train_labels_1~=1)',train_feats1(:,train_labels_1==1)',pBoost);


pTrain = {'maxDepth',15,'M',20,'minChild',1};
classifier =struct('tdata', forestTrain(train_feats1' ,(train_labels_1==1)'+1, pTrain));

%%

%sel_test = ~[fra_db.isTrain] & img_sel_score' & [fra_db.classID] ~= conf.class_enum.DRINKING;
sel_test = ~[fra_db.isTrain] & img_sel_score';
f_sel_test = find(sel_test);
img_scores = -inf*ones(size(f_sel_test));

[r,ir] = sort(img_scores,'descend');
for t =1:length(r)
    r(t)
    imgData = fra_db(f_sel_test(ir(t)));
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,imgData,curParams.roiParams);
    clf; imagesc2(I); pause
end

%%
for u = 1:1:length(f_sel_test)
    u
    %imgData = fra_db_train(t2+u);
    imgData = fra_db(f_sel_test(u));
    %     [regionFeats,imageFeats,selected_regions] = extract_all_features(conf,imgData,curParams);
    %     resPath = j2m(curParams.featsDir,imgData);
    %     L = load(resPath);
    %     I = L.imageFeats.I;
    [labels,features,ovps,is_gt_region] = collectFeaturesFromImg(conf,imgData,curParams);
    
    %     features = vl_homkermap(features,1);
    %     r = col(classifier.w(1:end-1)'*features);%+b;
    r = col(w'*features);%+b;
    img_scores(u) = max(r);
end

% 
%%

% save ~/storage/misc/bbox_predictor.mat classifier
load ~/storage/misc/bbox_predictor.mat
sel_train = [fra_db.isTrain] & img_sel_score';
sel_test = ~[fra_db.isTrain] & img_sel_score';
[boxes_data_train,image_data_train] = extractFeatsFromPredictedBoxes(conf,fra_db,sel_train,curParams,classifier);
[boxes_data_test,image_data_test] = extractFeatsFromPredictedBoxes(conf,fra_db,sel_test,curParams,classifier);

% save ~/storage/misc/img_and_box_data.mat boxes_data_train image_data_train boxes_data_test image_data_test
load ~/storage/misc/img_and_box_data.mat;
%%
T_ovp = .5;
[faceFeats_train,mouthFeats_train,imgFeats_train,img_label_train,boxFeats_train,box_labels_train,img_ind_train,ovps_train] = ...
    boxDataToFeatureMatrix(fra_db(sel_train),boxes_data_train,image_data_train,T_ovp);
% boxFeats_train(:,ovps_train<=T_ovp) = [];
% box_labels_train(ovps_train<=T_ovp) = [];

[faceFeats_test,mouthFeats_test,imgFeats_test,img_label_test,boxFeats_test,box_labels_test,img_ind_test,ovps_test] = ...
    boxDataToFeatureMatrix(fra_db(sel_test),boxes_data_test,image_data_test,T_ovp);
myNormalizeFun = @(x) normalize_vec(x);
% myNormalizeFun = @(x) x; 

faceFeats_train = myNormalizeFun(faceFeats_train);
faceFeats_test = myNormalizeFun(faceFeats_test);
imgFeats_train = myNormalizeFun(imgFeats_train);
imgFeats_test = myNormalizeFun(imgFeats_test);
mouthFeats_train = myNormalizeFun(mouthFeats_train);
mouthFeats_test = myNormalizeFun(mouthFeats_test);
allFeats_train = [faceFeats_train; imgFeats_train; mouthFeats_train];
allFeats_test = [faceFeats_test; imgFeats_test; mouthFeats_test];
boxFeats_train = myNormalizeFun(boxFeats_train);
boxFeats_test = myNormalizeFun(boxFeats_test);
balanceDirection = 0;

classifiers = struct('name',{},'feats_train',{},'feats_test',{},'classifier',{},'perf',{});
t = 1;
classifiers(t).name = 'face';
classifiers(t).feats_train = faceFeats_train;
classifiers(t).feats_test = faceFeats_test; t = t+1;
classifiers(t).name = 'mouth';
classifiers(t).feats_train = mouthFeats_train;
classifiers(t).feats_test = mouthFeats_test; t = t+1;
classifiers(t).name = 'mouth+face';
classifiers(t).feats_train = [mouthFeats_train;faceFeats_train];
classifiers(t).feats_test = [mouthFeats_test;faceFeats_test]; t = t+1;
classifiers(t).name = 'img';
classifiers(t).feats_train = imgFeats_train;
classifiers(t).feats_test = imgFeats_test; t = t+1;
classifiers(t).name = 'all';
classifiers(t).feats_train = allFeats_train;
classifiers(t).feats_test = allFeats_test; t = t+1;

% add hog features!!! 
% hogFeatsTrain = getImageStackHOG(subImgs(sel_train),[48 48]);
% hogFeatsTest = getImageStackHOG(subImgs(sel_test),[48 48]);
%
% classifiers(t).name = 'hogs';
% classifiers(t).feats_train = (hogFeatsTrain);
% classifiers(t).feats_test = (hogFeatsTest); t = t+1;
% 
% %
% classifiers(t).name = 'hogs+mouth+face';
% classifiers(t).feats_train =[mouthFeats_train;faceFeats_train;myNormalizeFun(hogFeatsTrain)];
% classifiers(t).feats_test = [mouthFeats_test;faceFeats_test;myNormalizeFun(hogFeatsTest)]; t = t+1;


%%

% 1. single-class classification
curClass = conf.class_enum.BRUSHING_TEETH;
classifier_choice = [3 4]
for it = 1:length(classifier_choice)
    t = classifier_choice(it)
    curClassifier = train_classifier_helper_2(img_label_train,classifiers(t).feats_train,curClass,balanceDirection);
%     curClassifier = train_classifier_helper_2(img_label_train,classifiers(t).feats_train,curClass,1);
    classifiers(t).classifier = curClassifier;
end

mm = ceil(sqrt(length(classifier_choice)));
nn = ceil(length(classifier_choice)/mm);

figure(1);
for it = 1:length(classifier_choice)
    t = classifier_choice(it)
    curClassifier = classifiers(t).classifier;
    curTestRes = curClassifier.w(1:end-1)'*(classifiers(t).feats_test);
    [test_scores,test_labels] = to_orig_test(curTestRes,fra_db,sel_test);
    classifiers(t).perf.test_scores = test_scores;
    classifiers(t).perf.test_labels = test_labels;
    subplot(mm,nn,it);
    vl_pr(2*(test_labels==curClass)-1,test_scores);setCurrentTitle([', ' classifiers(t).name]);
end

% 
classifier_boxes = train_classifier_helper_2(box_labels_train,boxFeats_train,curClass,balanceDirection);
test_res_boxes = classifier_boxes.w(1:end-1)'*(boxFeats_test);
test_res_boxes = reshape(test_res_boxes,[],nnz(sel_test));
[test_res_boxes,im] = max(test_res_boxes,[],1);
[test_scores_boxes,test_labels_boxes] = to_orig_test(test_res_boxes,fra_db,sel_test);
test_scores_indices = to_orig_test(im,fra_db,sel_test);


%%
final_scores = 5*classifiers(3).perf.test_scores + classifiers(4).perf.test_scores+1*test_scores_boxes;
testImages = fra_db(~[fra_db.isTrain]);
test_subs = subImgs(~[fra_db.isTrain]);
[u,iu] = sort(final_scores,'descend');
f_test = find(~[fra_db.isTrain]);
f_sel_test = iu;
displayImageSeries(conf,test_subs(iu),0,classifiers(3).perf.test_labels(iu),-1,f_test(iu));

fra_db(f_test(iu(33)))

vl_pr(2*(test_labels==curClass)-1,final_scores);setCurrentTitle([', ' classifiers(t).name]);
% 
%%
% classifier_face = train_classifier_helper_2(img_label_train,faceFeats_train,curClass,balanceDirection);
% test_res_face = classifier_face.w(1:end-1)'*(faceFeats_test);
% classifier_img = train_classifier_helper_2(img_label_train,imgFeats_train,curClass,balanceDirection);
% test_res_img = classifier_img.w(1:end-1)'*(imgFeats_test);
% classifier_mouth = train_classifier_helper_2(img_label_train,mouthFeats_train,curClass,balanceDirection);
% test_res_mouth = classifier_mouth.w(1:end-1)'*(faceFeats_test);
% [test_scores_mouth,test_labels] = to_orig_test(test_res_face,fra_db,sel_test);
% classifier_all = train_classifier_helper_2(img_label_train,allFeats_train,curClass,balanceDirection);
% test_res_all = classifier_all.w(1:end-1)'*(faceFeats_test);
% [test_scores_face,test_labels] = to_orig_test(test_res_face,fra_db,sel_test);

figure(1)
vl_pr(2*(test_labels==curClass)-1,test_scores_face);setCurrentTitle(', face');
[test_scores_mouth,test_labels] = to_orig_test(test_res_mouth,fra_db,sel_test);
figure(2)
vl_pr(2*(test_labels==curClass)-1,test_scores_mouth);setCurrentTitle(', mouth');

%
% test_res_boxes = reshape(test_res_boxes,[],nnz(sel_test));test_res_boxes  = test_res_boxes (1,:);
figure(3)
vl_pr(2*(test_labels==curClass)-1,test_scores_boxes);setCurrentTitle(', boxes');

figure(5)
[test_scores_img,test_labels] = to_orig_test(test_res_img,fra_db,sel_test);
vl_pr(2*(test_labels==curClass)-1,test_scores_img);setCurrentTitle(', img');

figure(6)
[test_scores_all,test_labels] = to_orig_test(test_res_all,fra_db,sel_test);
vl_pr(2*(test_labels==curClass)-1,test_scores_all);setCurrentTitle(', all');


%%
% 2. hierarchical classification: first apply the 4 classes vs everything
% else, then distinuish between these classes. 
curClass = [conf.class_enum.DRINKING conf.class_enum.SMOKING conf.class_enum.BLOWING_BUBBLES conf.class_enum.BRUSHING_TEETH];
classifier_face_h = train_classifier_helper_2(img_label_train,faceFeats_train,curClass,balanceDirection);
test_res_face_h = classifier_face_h.w(1:end-1)'*(faceFeats_test);
[test_scores_face_h,test_labels_h] = to_orig_test(test_res_face_h,fra_db,sel_test);
figure(1)
curTestLabels= 2*(ismember(test_labels,curClass))-1;
vl_pr(curTestLabels,test_scores_face_h);


for t = -3:.1:5
    t
figure(1)
vl_pr(2*(test_labels==curClass(1))-1,test_scores_face+test_scores_img+(test_scores_face_h>t));
pause(.1)
end
% vl_pr(2*(test_labels==curClass)-1,test_scores_face+1*test_scores_boxes)


%%
figure(4)

vl_pr(2*(test_labels==curClass)-1,test_scores_boxes+5*test_scores_face+0.1*test_scores_mouth);

%%
[u,iu] = sort(test_scores_face,'descend');
testImages = fra_db(~[fra_db.isTrain]);
f_sel_test = iu;

displayImageSeries(conf,testImages(iu),0,curTestLabels(iu),-1)

%%
[u,iu] = sort(test_scores_mouth,'descend');
sel_test

%% cluster images.

%% 
[IC,C] = kmeans2(faceFeats_train',5);balanceDirection = 0;
classifier_face_c = train_csvm(img_label_train,faceFeats_train,curClass,balanceDirection,C);
ws = cat(2,classifier_face_c.w);
ws = ws(1:end-1,:);
%%
test_res_face = ws'*(faceFeats_test);
D = l2(faceFeats_test',C);
[m,im] = min(D,[],2);

test_res_face =  test_res_face(sub2ind(size(test_res_face),im',1:size(test_res_face,2)));
%test_res_face = min(test_res_face(:,:),[],1);
[test_scores_face,test_labels] = to_orig_test(test_res_face,fra_db,sel_test);
figure(1)
vl_pr(2*(test_labels==curClass)-1,test_scores_face);

% per-cluster accuracy

inds_ = find(sel_test);
%cur_subset = sel_test;
cur_subset = false(size(fra_db));
cur_subset(inds_(im==u)) = 1;
for u = 1:size(C,1)
    test_res_face = ws'*(faceFeats_test);
    cur_scores = test_res_face(u,im==u);
    
    [test_scores_face,test_labels] = to_orig_test(test_res_face,fra_db,sel_test);
    figure(1)
    vl_pr(2*(test_labels==curClass)-1,test_scores_face);
end


outDir=  [];
maxPerCluster = 100;
[clusters,ims,inds] = makeClusterImages(subImgs(sel_train),C',IC,faceFeats_train,outDir,maxPerCluster);
mImage(ims);

%IC_test = assign_to_clusters(C,facef

displayImageSeries(conf,ims);

subImgs = {};
for u = 1:length(img_sel_score)
    u
    if (img_sel_score(u))
        resPath = j2m(curParams.featsDir,fra_db(u));
        L = load(resPath);
        I = L.imageFeats.I; 
        I = im2uint8(I);
        subImgs{u} = I;
    end
end
save ~/storage/misc/all_sub_imgs.mat subImgs

%%

D = l2(mouthFeats_train',mouthFeats_train');
[d,id_mouth] = sort(D,2,'ascend');
D = l2(faceFeats_train',faceFeats_train');
[d,id_face] = sort(D,2,'ascend');
train_subs = subImgs(sel_train);
%%

knn = 16;
for u = 520:10:length(train_subs)
    
    u
    figure(1)
    clf;
    subplot(2,2,1); imagesc2(train_subs{u});
    subplot(2,2,2); imagesc2(mImage(train_subs(id_face(u,2:knn+1))));
    subplot(2,2,3); imagesc2(mImage(train_subs(id_mouth(u,2:knn+1))));
    
    figure(2)
    
I = im2uint8(train_subs{u});
tic, bbs=edgeBoxes(I,model,opts); toc
bbs(:,3:4) = bbs(:,3:4)+bbs(:,1:2);
for t = 1:5 %size(bbs,1)
    clf; imagesc2(I); plotBoxes(bbs(t,:));
    title(num2str(bbs(t,end)));
    pause(.1)
    
end

%     pause
end

    
%%
% sel_test = ~[fra_db.isTrain] & img_sel_score';
%f_sel_test = find(sel_test);
%
% params.requiredKeypoints
mm = 2;
nn = 3;
for u = 1:1:length(f_sel_test)
    %imgData = fra_db_train(t2+u);
    f_sel_test(u)
    imgData = testImages(f_sel_test(u))
    if (imgData.classID==conf.class_enum.BRUSHING_TEETH),continue,end
    resPath = j2m(curParams.featsDir,imgData);
    L = load(resPath);
    I = L.imageFeats.I;
    [labels,features,ovps,is_gt_region] = collectFeaturesFromImg(conf,imgData,curParams);    
    % remove duplicate features
    features = features';
    [features,i_subset] = unique(features,'rows');
    labels = labels(i_subset);
    ovps = ovps(i_subset);
    is_gt_region = is_gt_region(i_subset);       
    r = adaBoostApply(features,classifier,[],[],8);
    rr = r;
    rr = normalise(rr);  
    
    [v,iv] = sort(rr,'descend');
    boxes = cat(1,L.regionFeats(i_subset).bbox);
    %     boxes = [boxes r];
    boxes = [boxes(iv(1:5),:) 5+r(iv(1:5))];
        
    [map,counts] = computeHeatMap(I,boxes,'sum');
    
    clf; subplot(mm,nn,1); imagesc2(I);
            
    plotBoxes(L.imageFeats.roiMouth);  %   title(num2str(imgData.raw_faceDetections.boxes(1,end)));
    subplot(mm,nn,2); imagesc2(map);
    subplot(mm,nn,3); imagesc2(sc(cat(3,map,I),'prob'));
    subplot(mm,nn,4); imagesc2(sc(cat(3,L.imageFeats.predictions.maskPredictionImage,I),'prob'));
    plotBoxes(L.imageFeats.kp_preds);
    [rois,roiBox,~,scaleFactor,roiParams] = get_rois_fra(conf,imgData,curParams.roiParams);
    plotBoxes(rois(1).bbox);    
    % also plot the "best" box
    bestBoxInd = test_scores_indices(f_sel_test(u));
    plotBoxes(boxes(bestBoxInd,:),'color','r');
%     
    title(num2str(max(r)));
    subplot(mm,nn,5); imagesc2(sc(cat(3,L.imageFeats.ucm,I),'prob'));
    subplot(mm,nn,6); imagesc2(cropper(I,L.imageFeats.roiMouth));
%     saveas(gcf, fullfile('/home/amirro/notes/images/2014_12_30/brush/',[sprintf('%05.0f_',u),imgData.imageID]));
    pause
end
%%
