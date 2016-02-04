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
%     rmpath(genpath('/home/amirro/code/3rdparty/attribute_code'));
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
    load ~/code/mircs/s40_fra_faces_d.mat
    load s40_fra
    s40_fra_orig = s40_fra_faces_d
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);
    fra_db = s40_fra;
    net = init_nn_netwclork();    
    nImages = length(s40_fra);
    top_face_scores = zeros(nImages,1);
    for t = 1:nImages
        top_face_scores(t) = max(s40_fra(t).raw_faceDetections.boxes(:,end));
    end
    min_face_score = 0;
    img_sel_score = col(top_face_scores > min_face_score);
    
    img_sel_score = col(top_face_scores > min_face_score)
    
%     img_sel_score = find(img_sel_score);
    fra_db = s40_fra(img_sel_score);
    top_face_scores_sel = top_face_scores(img_sel_score);
end

%% initialize parameters for different feature types.
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
stage_params(1).learning.classifierType = 'svm';
stage_params(1).learning.classifierParams.useKerMap = false;
dataDir = '~/storage/s40_fra_feature_pipeline_all';
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

imageLevelFeaturesPath = '~/storage/misc/imageLevelFeatures.mat';
if (exist(imageLevelFeaturesPath,'file'))
    load(imageLevelFeaturesPath);
else
    imageLevelFeatures = {};
    for t = 1:length(fra_db)
        t
        load(j2m('~/storage/s40_fra_feature_pipeline_partial_dnn_6_7',fra_db(t)),'moreData');
        imageLevelFeatures{t} = moreData.face_feats;
    end
    imageLevelFeatures = cat(2,imageLevelFeatures{:});
    save(imageLevelFeaturesPath,'imageLevelFeatures');
end

debugging = true;
stage1ClassifierPath = '~/storage/misc/stage_1_classifier.mat';
% "light" features for entire train set,
stage_1_feats_path = '~/storage/misc/stage_1_train_feats.mat';
if (exist(stage_1_feats_path,'file'))
    load(stage_1_feats_path);
else
    [feats1,labels1,ovps1] = collect_feature_subset(fra_db([fra_db.isTrain]),curParams);
    save(stage_1_feats_path,'feats1','labels1','ovps1');
end
[curFeats2,curLabels2,ovps2] = collect_feature_subset(fra_db(train_set),curParams);
classifierBaseDir = '~/storage/classifiers/stage1';ensuredir(classifierBaseDir);

for iSet = 1:length(sets)
    curTrainSet = sets{iSet};
    curValSet = sets{2-iSet+1};
    if (debugging)
        curTrainSet = curTrainSet(1:5:end);
        curValSet = curValSet(1:5:end);
    end
    
    classifier_1_data = struct('feats',{},'labels',{});    
    classifier_data = makeClassifierSet(fra_db,curTrainSet,curValSet,curParams,classes);
end
for iSet = 1:length(sets)
    curTrainSet = sets{iSet};
    curValSet = sets{2-iSet+1};
    if (debugging)
        curTrainSet = curTrainSet(1:5:end);
        curValSet = curValSet(1:5:end);
    end
    classifier_data = makeClassifierSet(fra_db,curTrainSet,curValSet,curParams,classes);
end

region_classifier_data_path = '~/storage/misc/region_classifiers.mat';
if (exist(region_classifier_data_path,'file'))
    load(region_classifier_data_path)
else
    debugging = false;
%     classifier_data = struct('iSet',{},'trainInds',{},'valInds',{},'classifiers',{},'val_results',{});
    
    for iSet = 1:length(sets)
        curTrainSet = sets{iSet};
        curValSet = sets{2-iSet+1};
        if (debugging)
            curTrainSet = curTrainSet(1:5:end);
            curValSet = curValSet(1:5:end);
        end        
        classifier_data = makeClassifierSet(fra_db,curTrainSet,curValSet,curParams,classes);
    end
    save(region_classifier_data_path,'classifier_data');
end

% train an overall region classifier

stage_1_boosting = stage_params(1);
stage_1_boosting.learning.classifierType = 'boosting';
stage_1_svm = stage_params(1);
stage_1_svm.learning.classifierType = 'svm';
stage_1_svm.learning.classifierParams.useKerMap = true;

boosting_region_classifier = makeClassifierSet(fra_db,find([fra_db.isTrain]),find(~[fra_db.isTrain]),stage_1_boosting,classes);
svm_region_classifier = makeClassifierSet(fra_db,find([fra_db.isTrain]),find(~[fra_db.isTrain]),stage_1_svm,classes);
stage_params_everything = defaultPipelineParams(true);
stage_params_everything.features.getAppearanceDNN = true;
stage_params_everything.features.getHOGShape = false;
stage_params_everything.dataDir = '~/storage/s40_fra_feature_pipeline_all_everything';
stage_params_everything.learning.classifierType = 'svm';
stage_params_everything.learning.classifierParams.useKerMap = true;

svm_region_classifier_everything = makeClassifierSet(fra_db,find([fra_db.isTrain]),find(~[fra_db.isTrain]),stage_params_everything,classes);

% now retain only the top few regions from each of the valitation results,
stage_1_res_dir = '~/storage/stage_1_subsets';
top_k_false = 100;
for ii = 1:length(classifier_data)
    prepareForNextStage(classifier_data(ii).val_results,...
        fra_db(classifier_data(ii).valInds),...
        stage_params(1).dataDir,...
        stage_1_res_dir,stage_params(1),top_k_false);
end

kk = findImageIndex(fra_db,'drinking_053.jpg');
kk = findImageIndex(fra_db,'smoking_213.jpg');
%[curFeats,curLabels] = collect_feature_subset(fra_db(k),stage_params(1));
v_results = applyToImageSet(conf,fra_db(k),classifiers,stage_params(1));
% set state 2 parameters
stage_params(2) = stage_params(1);
stage_params(2).features.getHOGShape = true;
stage_params(2).features.getAppearanceDNN = true;
stage_params(2).prevStageDir = '~/storage/stage_1_subsets';
stage_params(2).features.dnn_net = init_nn_network();
stage_params_dummy = stage_params(1);
extract_all_features(conf,fra_db(kk),stage_params_dummy);

%% extract dnn features only...
stage_params(2) = resetFeatures(stage_params(2),true);%resetFeatures(stage_params(2),false);
stage_params(2).features.getBoxFeats = 0;
stage_params(2).features.getAppearanceDNN = true;
stage_params(2).dataDir = '~/storage/s40_fra_feature_pipeline_stage_2';
stage_params(2).learning.classifierType = 'svm';
stage_params(2).learning.classifierParams.useKerMap = false;
my_train_set = train_set;
my_val_set = val_set;
region_classifier_data_path_2 = '~/storage/misc/region_classifiers_2.mat';
if (exist(region_classifier_data_path_2,'file'))
    load(region_classifier_data_path_2)
else
    stage_2_classifiers = {};
    stage_2_classifiers{1} = makeClassifierSet(fra_db,my_train_set,my_val_set,stage_params(2),classes);
    stage_2_classifiers{2} = makeClassifierSet(fra_db,my_val_set,my_train_set,stage_params(2),classes);
    save(region_classifier_data_path_2,'stage_2_classifiers');
end

% show results...
curRes = stage_2_classifiers{1};
[imgInds,scores] = summarizeResults(curRes);
[u,iu] = sort(scores,2,'descend');
displayImageSeries(conf,fra_db(imgInds(iu(2,:))));
iu1 = iu(2,:);
%%
for t = 1:length(curRes.valInds)
    
    imgInd = curRes.valInds(iu1(t));
%     k = iu1(t);
    curImgData =  switchToGroundTruth(fra_db(imgInd));
    if (curImgData.classID~=classes(2)),continue,end
    t
%     break
    disp(['face score = ' num2str(curImgData.raw_faceDetections.boxes(1,end))]);
%     if (fra_db(imgInd).classID~=9),continue,end
    resPath = j2m('~/storage/s40_fra_feature_pipeline_partial_dnn_6_7',curImgData);
    L = load(resPath,'feats','moreData','selected_regions');
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,curImgData,stage_params.roiParams); 
    iobj = find(strcmp({rois.name},'obj'))
    if none(iObj),continue,end
    rr = poly2mask2(rois(iobj).poly,size2(I)); 
    
    [ovps,ints,uns] = regionsOverlap({rr}, L.selected_regions);
    
    displayRegions(I,L.selected_regions,ovps,0,5);
    continue
    
    lm = loadDetectedLandmarks(conf,curImgData);
    lm = transformToLocalImageCoordinates(lm(:,1:4),scaleFactor,roiBox);
    moreFun = @(x) plotPolygons(boxCenters(lm),'g.','LineWidth',2);
    decision_values = curRes.val_results(iu1(t)).decision_values;
    clf;imagesc2(I); plotPolygons(boxCenters(lm),'g.','LineWidth',2); pause;
    displayRegions(I,L.selected_regions,decision_values(1,:),0,5);
    
    
    
%     displayRegions(I,L.selected_regions); 
end

%% the everything-based-classifier...
total_params = defaultPipelineParams(true);
total_params.features.getAppearanceDNN = true;
total_params.features.getHOGShape = true;
total_params.dataDir = '~/storage/s40_fra_feature_pipeline_all_everything';
total_params.learning.classifierType = 'svm';
total_params.learning.classifierParams.useKerMap = false;

train_indices = find([fra_db.isTrain]);
test_indices = find(~[fra_db.isTrain]);
% train_indices = train_indices(1:100:end);
% test_indices = test_indices(1:100:end);

% tried with random forest, didn't really work, trying svm + homkermap
maxNegFeats = 10;
classifier_all = makeClassifierSet(fra_db,train_indices,test_indices,total_params,classes,[],maxNegFeats)
save classifier_all classifier_all


classifiers = [classifiers{:}];
classifier_all = makeClassifierSet(fra_db,train_indices,test_indices,total_params,classes,classifiers);
save classifier_all classifier_all
% 
% theClassifiers = struct('tdata',{});
% for iClass = 1:length(classes)
%     theClassifiers(iClass).tdata = classifier_all.classifiers(:,iClass);
% end
classifier_all_corrected = makeClassifierSet(fra_db,train_indices,test_indices,total_params,classes,theClassifiers);
save classifier_all_corrected classifier_all_corrected

applyToImageSet(fra_db(test_set),classifier_all,total_params);

save classifier_all classifier_all

total_params.dataDir = '~/storage/face_only_feature_pipeline_all';
%%11/11/2014
res_all_faces = applyToImageSet(s40_fra_faces_d(~[s40_fra_faces_d.isTrain]),classifier_all.classifiers,total_params);
save res_all_faces res_all_faces
imageLevelFeaturesPath_new = '~/storage/misc/imageLevelFeatures_new.mat';
if (exist(imageLevelFeaturesPath_new,'file'))
    load(imageLevelFeaturesPath_new);
else
    imageLevelFeatures = {};
    for t = 1:length(fra_db)
        t
        load(j2m(total_params.dataDir,fra_db(t)),'moreData');
        imageLevelFeatures{t} = moreData.face_feats;
    end
    imageLevelFeatures = cat(2,imageLevelFeatures{:});
    save(imageLevelFeaturesPath_new,'imageLevelFeatures');
end
%%


% (run in parallel)
%% train a region based classifier on appearance
all_train_set = [fra_db.isTrain];
curParams = stage_params(2);
stage_params(2).dataDir = '~/storage/s40_fra_feature_pipeline_stage_2';
[curFeats,curLabels] = collect_feature_subset(fra_db(all_train_set(1:10)),curParams);
%classifiers{iSet} = train_region_classifier(conf,curFeats,curLabels,curClass,curParams);
stage_2_classifiers = {};
for iClass = 1:length(classes)
    classifiers{iClass} = train_region_classifier(conf,curFeats,curLabels,classes(iClass),curParams);
end

%classifier_data(iSet).val_results = applyToImageSet(conf,fra_db(curValSet),classifiers,curParams);

% region_classifiers_all = {};
% [curFeats,curLabels] = collect_feature_subset(fra_db(curTrainSet),curParams);    
% for iSet = 1
%     curTrainSet = sets{iSet};
%     curValSet = sets{2-iSet+1};    
%     [curFeats,curLabels] = collect_feature_subset(fra_db(curTrainSet),curParams);    
%     classifiers{iSet} = train_region_classifier(conf,curFeats,curLabels,curClass,curParams);                
% end
regional_test_results = applyToImageSet(conf,fra_db(test_set),classifiers{1},curParams);

%%
test_scores_region_2 = zeros(4,length(regional_test_results));
for t = 1:length(test_set)
    test_scores_region_2(:,t) = max(boosting_region_classifier.val_results(t).decision_values,[],2);
end

test_scores_region_svm = zeros(4,length(regional_test_results));
for t = 1:length(test_set)
    test_scores_region_svm(:,t) = max(svm_region_classifier.val_results(t).decision_values,[],2);
end

test_scores_region_all = zeros(4,length(classifier_all.val_results));
for t = 1:length(test_set)
    test_scores_region_all(:,t) = max(classifier_all.val_results(t).decision_values,[],2);
end
%%
AA  =load('~/storage/misc/all_s40_dnn_m_2028');
% all_s40_dnn = all_s40_dnn_verydeep;
% 
all_s40_dnn = AA.all_s40_dnn_m_2048;
myNormalizeFun = @(x) normalize_vec(x);
% myNormalizeFun  = @(x) x;
%%
% global_feats = myNormalizeFun([all_s40_dnn.full_feat]); % entire image
% global_feats_sel = global_feats(:,img_sel_score); % entire image, selection by score
feats_face = myNormalizeFun([imageLevelFeatures.global_17]); % face features (selection)
feats_mouth = myNormalizeFun([imageLevelFeatures.mouth_17]); % mouth features (selection)
feats_mouth_and_face = myNormalizeFun([[imageLevelFeatures.global_17];[imageLevelFeatures.mouth_17]]); % mouth features (selection)
%%
clear perf_ours perf_faces
% feats_test_face = feat_face(:,all_test);
% ([imageLevelFeatures(all_test).global]);
% feats_test_crop = (crop_feats_sel(:,all_test));
% feats_test_global = global_feats_sel(:,all_test);
results_with_global = {};
results_ours = {};
infos_ours_tries = {};

all_face_scores = zeros(length(classes),length(all_test));
all_region_scores = test_scores_region_all;





%%
%%
for iClass = 1:4
    curClass = classes(iClass);
    all_test = find(~[fra_db.isTrain]);
    %     curClass = conf.class_enum.READING;
    % find the optimal weight....
    %         classifier_face = train_classifier_helper(fra_db,feats_face,curClass);
    %         classifier_mouth = train_classifier_helper(fra_db,feats_mouth,curClass);
    
    classifier_face_and_mouth = train_classifier_helper(fra_db,feats_mouth_and_face,curClass);
    
    %
    %         test_res_face = classifier_face.w(1:end-1)'*(feats_face(:,all_test));
    %         test_res_mouth = classifier_mouth.w(1:end-1)'*(feats_mouth(:,all_test));
    test_res_face_and_mouth = classifier_face_and_mouth.w(1:end-1)'*(feats_mouth_and_face(:,all_test));    
    orig_test = ~[s40_fra.isTrain];
    orig_labels = [s40_fra.classID];
    orig_labels = 2*(orig_labels==curClass)-1;    
    orig_scores = -100*ones(size(s40_fra));
    orig_scores = orig_scores + rand(size(orig_scores ));
    %
    all_face_scores(iClass,:) = test_res_face_and_mouth;
    %         all_region_scores(iClass,:) =
    orig_scores([fra_db(all_test).imgIndex]) = test_res_face_and_mouth;
    [perf_faces(iClass).recall, perf_faces(iClass).precision, perf_faces(iClass).info] = ...
        vl_pr(orig_labels(orig_test),orig_scores(orig_test));
    %
    %      totalScores = test_res_face+.5*test_res_mouth;
    %    curClass = classes(iClass)
    %     totalScores = test_scores_region_all(iClass,:);
    
    %     vl_pr(orig_labels(orig_test),orig_scores(orig_test));
    %
    %     all_labels = 2*([fra_db(all_test).classID]==curClass)-1;
end
%

%
%     infos_ours = [perf_ours.info];
infos_ours_single_svm = [perf_faces.info];
%     ap_infos_ours = [infos_ours.ap]
ap_infos_ours_single_svm = [infos_ours_single_svm.ap]
infos_ours_tries{end+1} = ap_infos_ours;
%%


mixture_coeffs = 0:.1:.5;
% mixture_coeffs = .1
combined_aps = zeros(length(classes),length(mixture_coeffs));
for ii = 1:length(mixture_coeffs)
    for iClass = 1:4
        curClass = classes(iClass);
        orig_test = ~[s40_fra.isTrain] & cur_score_sel;
        orig_labels = [s40_fra.classID];
        orig_labels = 2*(orig_labels==curClass)-1;
        orig_scores = -20*ones(size(s40_fra));
        orig_scores = orig_scores + rand(size(orig_scores ));        
        orig_scores([fra_db(all_test).imgIndex]) = all_face_scores(iClass,:);
        [perf_faces(iClass).recall, perf_faces(iClass).precision, perf_faces(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        orig_scores([fra_db(all_test).imgIndex]) = all_region_scores(iClass,:);
        [perf_regions(iClass).recall, perf_regions(iClass).precision, perf_regions(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        orig_scores([fra_db(all_test).imgIndex]) = all_face_scores(iClass,:)+mixture_coeffs(ii)*all_region_scores(iClass,:);
        [perf_mixed(iClass).recall, perf_mixed(iClass).precision, perf_mixed(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        combined_aps(iClass,ii) = perf_mixed(iClass).info.ap;        
    end
end



info_faces =[perf_faces.info];
ap_faces = [info_faces.ap]
info_regions =[perf_regions.info];
ap_regions = [info_regions.ap]
plot(mixture_coeffs, combined_aps')
legend(conf.classes(classes));
% info_mixed =[perf_mixed.info];
% ap_mixed = [info_mixed.ap]

%% check how, if we limit ourselves to images where  faces are detected above a certain threshold only, 
%% the a.p changes
cur_min_score = -inf;
face_score_thresholds = [-inf 0:.2:3]
%mixture_coeffs = 0:.1:.5;
mixture_coeffs = .1;
combined_aps = zeros(length(classes),length(face_score_thresholds));
nPosSamples = zeros(length(classes),length(face_score_thresholds));
for ii = 1:length(face_score_thresholds)
    cur_face_thresh = face_score_thresholds(ii);
    cur_score_sel = row(top_face_scores > cur_face_thresh);
    for iClass = 1:4
        curClass = classes(iClass);
        orig_test = ~[s40_fra.isTrain] & cur_score_sel;
        orig_labels = [s40_fra.classID];
        orig_labels = 2*(orig_labels==curClass)-1;
        nPosSamples(iClass,ii) = sum(orig_labels(orig_test)==1)/sum(orig_labels(~[s40_fra.isTrain])==1);
        
        orig_scores = -20*ones(size(s40_fra));
%         orig_scores = orig_scores + rand(size(orig_scores ));        
        orig_scores([fra_db(all_test).imgIndex]) = all_face_scores(iClass,:);
        [perf_faces(iClass).recall, perf_faces(iClass).precision, perf_faces(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        orig_scores([fra_db(all_test).imgIndex]) = all_region_scores(iClass,:);
        [perf_regions(iClass).recall, perf_regions(iClass).precision, perf_regions(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        orig_scores([fra_db(all_test).imgIndex]) = all_face_scores(iClass,:)+mixture_coeffs*all_region_scores(iClass,:);
        [perf_mixed(iClass).recall, perf_mixed(iClass).precision, perf_mixed(iClass).info] = ...
            vl_pr(orig_labels(orig_test),orig_scores(orig_test));        
        combined_aps(iClass,ii) = perf_mixed(iClass).info.ap;        
    end
end
%%
info_faces = [perf_faces.info];
ap_faces = [info_faces.ap]
info_regions =[perf_regions.info];
ap_regions = [info_regions.ap]
figure(1);clf; 
plot(face_score_thresholds, combined_aps'); hold on;
plot(face_score_thresholds, mean(nPosSamples),'k');
legend_str=  conf.classes(classes);
legend_str{end+1} = '% pos samples';
legend(legend_str);
% figure(2); clf;
% figure,plot(face_score_thresholds,nPosSamples');
% legend(conf.classes(classes));

%%
 plot(all_face_scores(1,:));
 hold on; plot(top_face_scores(orig_test),'g')