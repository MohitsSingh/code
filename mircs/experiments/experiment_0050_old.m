

%% 21/09/2014
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
    [learnParams,conf] = getDefaultLearningParams(conf,1024);
    load fra_db.mat;
    all_class_names = {fra_db.class};
    class_labels = [fra_db.classID];
    classes = unique(class_labels);
    % make sure class names corrsepond to labels....
    [lia,lib] = ismember(classes,class_labels);
    classNames = all_class_names(lib);
    isTrain = [fra_db.isTrain];
    normalizations = {'none','Normalized','SquareRoot','Improved'};
    %     addpath('/home/amirro/code/3rdparty/logsample/');
    addpath(genpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained'));install
    addpath(genpath('/home/amirro/code/3rdparty/cvpr14_saliency_code'));
    addpath(genpath('/home/amirro/code/3rdparty/attribute_code'));
    addpath(genpath('/home/amirro/code/3rdparty/seg_transfer'));
    % for edge boxes
    model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;
    [learnParams,conf] = getDefaultLearningParams(conf,256);
    featureExtractor = learnParams.featureExtractors{1};
    featureExtractor.bowConf.featArgs = {'Step',1};
    featureExtractor.bowConf.bowmodel.numSpatialX = [1 2];
    featureExtractor.bowConf.bowmodel.numSpatialY = [1 2];
    initialized = true;
    conf.get_full_image = true;
    load s40_fra
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);
    fra_db = s40_fra;
    net = init_nn_network();
    
    nImages = length(s40_fra);
    top_face_scores = zeros(nImages,1);
    for t = 1:nImages
        top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
    end
    min_face_score = 0;
    img_sel_score = col(top_face_scores > min_face_score);
    fra_db = s40_fra(img_sel_score);
end

%% restrict fra_db to exclude class 5 (phone)
% fra_db_orig = fra_db;
% % % % 
% % % % in_fra_db = false(size(s40_fra));
% % % % for t = 1:length(s40_fra)
% % % %     in_fra_db(t) = any(s40_fra(t).indInFraDB) && s40_fra(t).indInFraDB~=-1;
% % % % end
% % % % 
% % % % %fra_db = fra_db_orig([fra_db_orig.classID]~=5);
% % % % fra_db = s40_fra(in_fra_db);
% % % % isTrain = [fra_db.isTrain];


%% initialize parameters for different feature types.
% params = initAlgParams;
params = defaultPipelineParams();

%%

train_set = find([fra_db.isTrain]);
train_set = train_set(1:2:end);
test_set = find(~[fra_db.isTrain]);
val_set = setdiff(find([fra_db.isTrain]),train_set);

%%
% aggregate features and labels, from valid images only, for now
% valids = {};
% u = {};
% validImages = cellfun(@(x) x.valid,all_train_feats);
% train a set of classifiers
clear stage_params
stage_params(1)= defaultPipelineParams();
stage_params(1).learning.classifierType = 'boosting';
stage_params(1).learning.classifierParams.useKerMap = false;

dataDir = '~/storage/s40_fra_feature_pipeline_all';
stage_params(1).dataDir = dataDir;

% [labels,features,ovps] = collectFeatures(conf,all_train_feats,stage_params(1).params.features);
iClass = 1;
classes = [conf.class_enum.DRINKING conf.class_enum.SMOKING conf.class_enum.BLOWING_BUBBLES conf.class_enum.BRUSHING_TEETH];
curClass = classes(iClass);
curSet = val_set;
valParams = stage_params(1);
valParams.features.getAppearance = true;
top_k_false = 100;
sets = {train_set,val_set};
% set_results = {};
classifiers = {};
curParams = stage_params(1);
ind_in_orig = {};
debugging = false;
clear all_results;

for iSet = 1:length(sets)
    curTrainSet = sets{iSet};
    curValSet = sets{2-iSet+1};
    if (debugging)
        curTrainSet = curTrainSet(1:30:end);
        curValSet = curValSet(1:30:end);
    end
    [curFeats,curLabels] = collect_feature_subset(fra_db(curTrainSet),curParams);
%     curParams.features.getHOGShape = true;
%     curParams.features.getAppearanceDNN = true;
%     extract_all_features(conf,fra_db(curTrainSet(1)),curParams);    
    classifiers{iSet} = train_region_classifier(conf,curFeats,curLabels,curClass,curParams);
    curResults = applyToImageSet(conf,fra_db(curValSet),classifiers{iSet},curParams)
    for u = 1:length(curResults)
        all_results(curValSet(u)) = curResults(u);
    end
end

% extract appearance of mouth & face regions
for iSet = 1:length(sets)
    curTrainSet = sets{iSet};
    curValSet = sets{2-iSet+1};
    if (debugging)
        curTrainSet = curTrainSet(1:30:end);
        curValSet = curValSet(1:30:end);
    end
    train_feats = {};
    train_labels = {};
    
    for u = 1:length(curTrainSet)
        u
        imgData = fra_db(curTrainSet(u));
        resPath = j2m(curParams.dataDir,imgData);
        L = load(resPath,'moreData'); % don't need the segmentation here...
        curRoi = poly2mask2(round(makeSquare(L.moreData.roiMouth)),size2(L.moreData.I));
        train_feats{end+1} = featureExtractor.extractFeatures(L.moreData.I,curRoi,'normalization','Improved');
        train_labels{end+1} = imgData.classID;
%         train_feats{end+1} = featureExtractor.extractFeatures(flip_image( L.moreData.I),...
%             flip_image(curRoi));
%         train_labels{end+1} = imgData.classID;
        %         [labels,features,ovps,is_gt_region] = collectFeatures(L,params.features);
    end
    train_feats = cat(2,train_feats{:});
    train_labels = cat(2,train_labels{:});
    val_feats = {};
    val_labels = {};    
    for u = 1:length(curValSet)
        u
        imgData = fra_db(curValSet(u));
        resPath = j2m(curParams.dataDir,imgData);
        L = load(resPath,'moreData'); % don't need the segmentation here...
        curRoi = poly2mask2(round(makeSquare(L.moreData.roiMouth)),size2(L.moreData.I));
        val_feats{end+1} = featureExtractor.extractFeatures(L.moreData.I,curRoi,'normalization','Improved');
        val_labels{end+1} = imgData.classID;
%         train_feats{end+1} = featureExtractor.extractFeatures(flip_image( L.moreData.I),...
%             flip_image(curRoi));
%         train_labels{end+1} = imgData.classID;
        %         [labels,features,ovps,is_gt_region] = collectFeatures(L,params.features);
    end
    
    find(cellfun(@isempty,val_feats))
    for t = 1:size(val_feats,2)
        if (isempty(val_feats{t}))
            val_feats{t} = NaN(size(train_feats(:,1)));
        end
    end
    val_feats1 = cat(2,val_feats{:});
    
    val_labels1 = cat(2,val_labels{:});
    
    %%
    
    subImgsTrain = {};
    train_labels = {};
    imgsTrain = {};
    for u = 1:length(curTrainSet)
        u
        imgData = fra_db(curTrainSet(u));
        resPath = j2m(curParams.dataDir,imgData);
        L = load(resPath,'moreData'); % don't need the segmentation here...
%         m = cropper(L.moreData.I,round(makeSquare(L.moreData.roiMouth)));
%         prm = struct('hasChn',true);
%         IJ = jitterImage(m,'nTrn',7,'mTrn',3,'maxn',10,'hasChn',true);
%         IJ = imageStackToCell(IJ);
        
        subImgsTrain{end+1} = cropper(L.moreData.I,round(makeSquare(L.moreData.roiMouth)));
        imgsTrain{end+1} = L.moreData.I;
        train_labels{end+1} = imgData.classID;
%         subImgsTrain{end+1} = flip_image(subImgsTrain{end});
%         train_labels{end+1} = imgData.classID;
        %         subImgs{end+1} = flip_img(subImgs{end});
    end
        
    train_labels = cat(2,train_labels{:});    
    train_feats_dnn = extractDNNFeats(subImgsTrain,net);    
    train_feats_dnn_global = extractDNNFeats(imgsTrain,net);
    
    subImgsVal = {};
    imgsVal = {};
    val_labels = {};
    for u = 1:length(curValSet)
        u
        imgData = fra_db(curValSet(u));
        resPath = j2m(curParams.dataDir,imgData);
        L = load(resPath,'moreData'); % don't need the segmentation here...
        subImgsVal{end+1} = cropper(L.moreData.I,round(makeSquare(L.moreData.roiMouth)));
        imgsVal{end+1} = L.moreData.I;
        val_labels{end+1} = imgData.classID;
        
        %         subImgs{end+1} = flip_img(subImgs{end});
    end
    val_labels = cat(2,val_labels{:});

    val_feats_dnn = extractDNNFeats(subImgsVal,net);
    val_feats_dnn_global = extractDNNFeats(imgsVal,net);
    %%
    nLayer = 17;
    val_feats_dnn = squeeze(res_val(nLayer).x);
%     val_feats = vl_homkermap(val_feats,1);
%     imagesc(val_feats);
    train_feats_dnn = squeeze(res_train(nLayer).x);
%     train_feats = vl_homkermap(train_feats,1);
%%
    curClass = classes(3);
    clf    
    [posFeats,negFeats] = splitFeats(double(train_feats_dnn(:,1:1:end)),train_labels(:,1:1:end)==curClass);        
    mouth_classifier_dnn = train_classifier_pegasos(posFeats,negFeats,0,false);
    val_res_dnn = mouth_classifier_dnn.w(1:end-1)'*val_feats_dnn;
    vl_pr(2*([fra_db(curValSet).classID]==curClass)-1,val_res_dnn)    
    
    %% global...(from all face regions)
    clf    
    [posFeats,negFeats] = splitFeats(double(train_feats_dnn_global(:,1:1:end)),train_labels(:,1:1:end)==curClass);        
    mouth_classifier_dnn_global = train_classifier_pegasos(posFeats,negFeats,0,false);
    val_res_dnn_global = mouth_classifier_dnn_global.w(1:end-1)'*val_feats_dnn_global;
    vl_pr(2*([fra_db(curValSet).classID]==curClass)-1,val_res_dnn_global) 
    
%%
%     train_feats = cellfun2(@(x) col(fhog2(im2single(x),4)), subImgsTrain);train_feats =cat(2,train_feats {:});
%     val_feats = cellfun2(@(x) col(fhog2(im2single(x),4)), subImgsVal);val_feats =cat(2,val_feats {:});
%      val_feats = vl_homkermap(val_feats,1);
%      train_feats = vl_homkermap(train_feats,1);
    %%
% %     curClass = classes(1);
%     clf
%     [posFeats,negFeats] = splitFeats(train_feats,train_labels==curClass);
%     mouth_classifier = train_classifier_pegasos(posFeats,negFeats,0,false);
%     val_res = mouth_classifier.w(1:end-1)'*val_feats1;
%     vl_pr(2*([fra_db(curValSet).classID]==curClass)-1,val_res)   
    
    %%
    finalScore = 1*val_res+val_res_dnn+val_res_dnn_global;
    finalScore(isnan(finalScore)) = -inf;
    vl_pr(2*([fra_db(curValSet).classID]==curClass)-1,finalScore)
    %%
    [r,ir] = sort(val_res,'descend');
    %%
    
%     finalScore = val_
    [r,ir] = sort(finalScore,'descend');
    
    for u = 1:length(curValSet)
        u
        imgData = fra_db(curValSet(ir(u)));
        resPath = j2m(curParams.dataDir,imgData);
        L = load(resPath,'moreData'); % don't need the segmentation here...
        clf; subplot(1,2,1);imagesc2(L.moreData.I);
        subplot(1,2,2);imagesc2(L.moreData.saliency.sal);
        pause
    end
    %%
    val_labels = cat(2,val_labels{:});
    [curFeats,curLabels] = collect_feature_subset(fra_db(curTrainSet),curParams);
    classifiers{iSet} = train_region_classifier(conf,curFeats,curLabels,curClass,curParams);
    curResults = applyToImageSet(conf,fra_db(curValSet),classifiers{iSet},curParams)
    for u = 1:length(curResults)
        all_results(curValSet(u)) = curResults(u);
    end
end


%%
% prepare stage 1 results for feature calculation
stage_1_res_dir = '~/storage/stage_1_subsets';
top_k_false = 100;
prepareForNextStage(all_results,fra_db,stage_params(1).dataDir,stage_1_res_dir,valParams,top_k_false)
% calculate...
% collect results for training data
stage2Dir = '~/storage/s40_fra_feature_pipeline_stage_2';
stage_params(2) = stage_params(1);
stage_params(2).dataDir = stage2Dir;
stage_params(2) = resetFeatures(stage_params(2),false);
stage_params(2).features.getAttributeFeats = true;
stage_params(2).learning.classifierType = 'rand_forest';
[curFeats,curLabels] = collect_feature_subset(fra_db([train_set val_set]),stage_params(2));
nans = any(isnan(curFeats),1);
% train a classifer with all features involved
[posFeats,negFeats] = splitFeats(curFeats(:,~nans),curLabels(~nans)==curClass);

normalizeFeats = false;
% stage_params(2).learning.classifierType = 'boosting';
stage_2_classifier = train_region_classifier(conf,curFeats,curLabels,curClass,stage_params(2));

% collect features, apply classifier
% [curFeats,curLabels] = collect_feature_subset(fra_db(test_set),curParams);
% prepare stage 1 results for feature calculation on test images
test_results_1 = applyToImageSet(conf,fra_db(test_set),classifiers{1},curParams);
prepareForNextStage(test_results_1,fra_db(test_set),stage_params(1).dataDir,stage_1_res_dir,valParams,top_k_false)
% ....run calculation...

% run on test images as well...
% all_results = cat(2,set_results{:});
%%
stage2Dir = '~/storage/s40_fra_feature_pipeline_stage_2';
%[feats,labels] = loadStageResults(conf,all_results,fra_db,stage2Dir);
stage_params(2).dataDir = stage2Dir;
stage_params(2) = stage_params(1);
stage_params(2).features.getAppearance = true;
max_neg_to_keep = 30;
[curFeats,curLabels] = collect_feature_subset(fra_db([train_set val_set]),stage_params(2),max_neg_to_keep);
classifier_2 = train_region_classifier(conf,curFeats,curLabels,curClass,stage_params(2))
% collect some results from the
test_subset = ~[fra_db.isTrain] & [fra_db.classID]==9;
test_results = applyToImageSet(conf,fra_db(test_set),classifier_2,stage_params(2));

%%
%%
test_set_results = {};
for u = 1:length(val_set)
    max_neg_to_keep = inf;
    k = val_set(u)
        if (fra_db(k).classID~=classes(1)),continue,end
    resPath1 = j2m(stage_params(1).dataDir,fra_db(k));
    L = load(resPath1,'feats','moreData','selected_regions');
    %     [featStruct.feats,featStruct.moreData,selected_regions] = extract_all_features(conf,fra_db(k),...
    %         stage_params(2),L.moreData,L.selected_regions);
    [labels, features,all_ovps,is_gt_region,orig_inds] = collectFeatures(L,stage_params(1).features);
    %load(resPath,'feats','moreData','masks'); % don't need the segmentation here...
    %     [curFeats,curLabels] = collect_feature_subset(fra_db(k),stage_params(2),max_neg_to_keep);
    
    
    f = apply_region_classifier(classifiers{1},features,stage_params(1));
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k),params.roiParams);
    displayRegions(I,L.selected_regions,f,0,5);
    
    
% %     f = apply_region_classifier(classifiers{1},features,stage_params(1))+...
% %         apply_region_classifier(classifiers{2},features,stage_params(1));
% %     test_set_results{k} = f;
    %     [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k),params.roiParams);
end
%%

m = -inf(size(fra_db));
f = find(~isTrain);
for t = 1:length(test_set)
    m(f(t)) = max(test_set_results{f(t)});
end

figure,vl_pr(2*([fra_db(~isTrain).classID]==curClass)-1,m(~isTrain));
[z,iz] = sort(m(~isTrain),'descend');
for r = 1:length(test_set)
    max_neg_to_keep = inf;
    k = test_set(iz(r));
    %     if (fra_db(k).classID~=9),continue,end
    resPath1 = j2m(stage_params(1).dataDir,fra_db(k));
    
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k),params.roiParams);
    clf; imagesc2(I); title(num2str(z(r)));pause;continue
    
    L = load(resPath1);
    %     [featStruct.feats,featStruct.moreData,selected_regions] = extract_all_features(conf,fra_db(k),...
    %         stage_params(2),L.moreData,L.selected_regions);
    [labels, features,all_ovps,is_gt_region,orig_inds] = collectFeatures(L,stage_params(1).features);
    %load(resPath,'feats','moreData','masks'); % don't need the segmentation here...
    %     [curFeats,curLabels] = collect_feature_subset(fra_db(k),stage_params(2),max_neg_to_keep);
    f = apply_region_classifier(classifiers{1},features,stage_params(1))+...
        apply_region_classifier(classifiers{2},features,stage_params(1));
    
end

test_set_results = {};
%%
for u = 1:length(test_set)
    max_neg_to_keep = inf;
    k = test_set(u);
    if (fra_db(k).classID~=9),continue,end
    resPath1 = j2m(stage_params(1).dataDir,fra_db(k));
    L = load(resPath1);
    %     [featStruct.feats,featStruct.moreData,selected_regions] = extract_all_features(conf,fra_db(k),...
    %         stage_params(2),L.moreData,L.selected_regions);
    [labels, features,all_ovps,is_gt_region,orig_inds] = collectFeatures(L,stage_params(1).features);
    %load(resPath,'feats','moreData','masks'); % don't need the segmentation here...
    %     [curFeats,curLabels] = collect_feature_subset(fra_db(k),stage_params(2),max_neg_to_keep);
    f = apply_region_classifier(classifiers{1},features,stage_params(1))+...
        apply_region_classifier(classifiers{2},features,stage_params(1));
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(k),params.roiParams);
    
    % end
    %     clf;
    %     disp('stage 1');
    %     displayRegions(I,L.selected_regions,f,0,1);continue
    % continue;
    %     break
    [u,iu] = sort(f,'descend');
    selected_regions = L.selected_regions(iu(1:100));
    u = u(1:5);
    stage_params(2).get_gt_regions = false;
    [featStruct.feats,featStruct.moreData,selected_regions] = ...
        extract_all_features(conf,fra_db(k),stage_params(2),L.moreData,selected_regions);
    
    [labels, features,all_ovps,is_gt_region,orig_inds] = collectFeatures(featStruct,stage_params(2).features);
    f = apply_region_classifier(stage_2_classifier,features,stage_params(2));
    disp('stage 2');
    displayRegions(I,selected_regions,f(:),0,1);
end


%% try some stuff on the phrasal recognition dataset...
baseDir = '/home/amirro/storage/data/UIUC_PhrasalRecognitionDataset/VOC3000/JPEGImages';
cacheTag = '~/storage/phrasal_rec';
funs = struct('tag',{},'fun',{});
funs(1).tag = 'faces';
funs(1).fun = 'fra_faces_baw_new';
funs(2).tag = 'face_landmarks';
funs(2).fun = 'my_facial_landmarks_new';
funs(3).tag = 'seg';
funs(3).fun = 'fra_face_seg_new';
funs(4).tag = 'obj_pred';
funs(4).fun = 'fra_obj_pred_new';
funs(5).tag = 'feat_pipeline_r';
funs(5).fun = 'fra_feature_pipeline_new';
funs(6).tag = 'ELSD';
funs(6).fun = 'run_elsd';

% run a
for t = 1:length(funs)
    funs(t).outDir = fullfile(cacheTag,funs(t).tag);
end
pipelineStruct = struct('baseDir',baseDir,'funs',funs);

%%
[paths,names] = getAllFiles(baseDir,'.jpg');
%%
all_class_scores = -inf(length(names),5);
%%
LL = cell(size(paths));
%%
for iImg =1:length(paths)
    iImg
    if (isempty(LL{iImg}))
        % load the basic features
        resPath = j2m(funs(5).outDir,names{iImg});
        if (~exist(resPath))
            continue
        end
        %     break
        L = load(resPath);
        LL{iImg} = L;
    end
end
%%
for iImg = 1:length(paths)
    iImg
    L = LL{iImg};
    if (isempty(L)),continue,end
    
    for iStage = 1%:length(stage_params)
        curParams = stage_params(iStage);
        featData = L.res.featData;
        [~,currentFeatures] = collectFeatures({featData},curParams.features);
        for iClass = 1
            if (strcmp(classifierType ,'rand_forest'))
                [hs,probs] = forestApply( currentFeatures', stage_forests{1});
                decision_values = probs(:,2);
            else
                decision_values = adaBoostApply(currentFeatures',obj(1).model,[],[],8);
            end
            %             f = adaBoostApply(currentFeatures',obj(iClass).model,[],[],8);
            all_class_scores(iImg,iClass) = max(decision_values(:));
            %         all_all_class_scores{u,1} = decision_values;
        end
    end
end

%%

[r,ir] = sort(all_class_scores(:,1),'descend');
% curSet = test_set;
for ff =1:length(paths)
    ff
    f = ir(ff);
    % load the basic features
    resPath = j2m(funs(5).outDir(1:end-2),names{f});
    if (~exist(resPath,'file'))
        warning ('file doesn''t exist - skipping');
        continue
    end
    faceScore = LL{f}.res.curImageData.faceScore
    if (faceScore < 1),continue,end
    L = load(resPath);
    featData = L.res.featData;
    curImgData = L.res.curImageData;
    
    for iStage = 1%:length(stage_params)
        curParams = stage_params(iStage);
        [~,currentFeatures] = collectFeatures({featData},curParams.features);
        if (strcmp(classifierType ,'rand_forest'))
            [hs,probs] = forestApply( currentFeatures', stage_forests{1});
            decision_values = probs(:,2);
        else
            decision_values = adaBoostApply(currentFeatures',obj(1).model,[],[],8);
        end
        %             f = adaBoostApply(currentFeatures',ob
        %                 [r,probs] = forestApply( single(currentFeatures)', stage_forests{iStage});
        
        [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImgData,params.roiParams);
        curMasks = {featData.feats.mask};
        %         curParams.debug = true;
        %         curParams1 = defaultPipelineParams(false);
        %         curParams1.debug = true;
        %         curParams1.pipelineParams = pipelineStruct.funs;
        extract_all_features_new(conf,curImgData,curParams1,featData);
        
        figure(2);%
        clf
        displayRegions(I,curMasks,decision_values,0,5);
        %         featData.segmentation.candidates.masks = curMasks(ik(1:100));
        continue;
        %%
        [zz,izz] = sort(probs,'descend');
        featData.segmentation.candidates.masks = curMasks(izz(1:10));
        featData = extract_all_features(conf,fra_db(u), stage_params(iStage+1),featData);
        [k,ik] = sort(decision_values,'descend');
        %%
        %         for tt = 1:10
        %             V = imfilter(im2double(curMasks{ik(tt)}),fspecial('gauss',[41 41],19));
        %             V = normalise(V);
        %             seg = st_segment(I,V,.5,3);
        %             displayRegions(I,seg);
        %             %gc_segResult = getSegments_graphCut_2(I,curMasks{ik(tt)},[],1);
        %             pause;
        %         end
        %         [regions,ovp,sel_] = chooseRegion(I,curMasks,.5);
        %         displayRegions(I,regions,ovp)
        
        %%
    end
    %     break;
end
