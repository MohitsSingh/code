
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
    
    load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
    load ~/storage/mircs_18_11_2014/s40_fra
    
    s40_fra_orig = s40_fra_faces_d
    params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
    load('~/storage/mircs_18_11_2014/allPtsNames','allPtsNames');
    [~,~,reqKeyPointInds] = intersect(params.requiredKeypoints,allPtsNames);
    fra_db = s40_fra_faces_d;


end



%%

classes = [conf.class_enum.DRINKING conf.class_enum.SMOKING conf.class_enum.BLOWING_BUBBLES conf.class_enum.BRUSHING_TEETH];

test_scores_region_all = -inf(4,length(fra_db));
bad_imgs = false(size(fra_db));
resultsDir = '~/storage/s40_fra_feature_pipeline_partial_dnn_6_7';
resultsDir = '~/storage/face_only_features_lite';
imageLevelFeatures = {};
for t = 1:length(fra_db)
    t
%     try 
        L = load(j2m(resultsDir,fra_db(t)));
%     catch e
%         '!'
%         delete(j2m(resultsDir,fra_db(t)))
%         continue
%     end 
    if (isempty(L) || isempty(fieldnames(L))  || isempty(L.moreData))
        bad_imgs(t) = true;
        continue;
    end
%     test_scores_region_all(:,t) = max(L.classificationResult.decision_values,[],2);
    imageLevelFeatures{t} = L.moreData.face_feats;
end
%     imageLevelFeatures = cat(2,imageLevelFeatures{:});

save('~/storage/misc/imageLevelFeatures_faces.mat','imageLevelFeatures');

find(bad_imgs)
examineImg(conf,fra_db(7936))
imageLevelFeatures{1}
find_bad = find(bad_imgs)
for f = find_bad
    imageLevelFeatures{f} = imageLevelFeatures{1};
end

imageLevelFeatures = cat(2,imageLevelFeatures{:});
%%

myNormalizeFun = @(x) normalize_vec(x);
%
feats_face = myNormalizeFun([imageLevelFeatures.global_17]); % face features (selection)
feats_mouth = myNormalizeFun([imageLevelFeatures.mouth_17]); % mouth features (selection)
feats_mouth_and_face = myNormalizeFun([[imageLevelFeatures.global_17];[imageLevelFeatures.mouth_17]]); % mouth features (selection)
%%
clear perf_ours perf_faces
results_ours = {};
infos_ours_tries = {};
all_face_scores = zeros(length(classes),length(all_test));
all_region_scores = test_scores_region_all;
%%
face_det_scores = [fra_db.faceScore];
%%
T_face = -5:.1:5;
all_aps = zeros(4,length(T_face));
%%
all_classes = [fra_db.classID];
%[~,is_fra] = ismember(classes,all_classes);
is_fra = false(size(fra_db));
for t = 1:length(classes)
    is_fra = is_fra | all_classes==classes(t);
end




%%
for iClass = 1:4
    curClass = classes(iClass);
    all_test = find(~[fra_db.isTrain]);
     
    toNormalize = false;
    classifier_face_and_mouth = train_classifier_helper(fra_db,feats_mouth_and_face,curClass,toNormalize,~bad_imgs);% & (top_face_scores' > 0));
    
%     classifier_face = train_classifier_helper(fra_db,feats_face,curClass,toNormalize,~bads_imgs);
%     classifier_mouth = train_classifier_helper(fra_db,feats_mouth,curClass,toNormalize,~bads_imgs);       
%
    test_res_face_and_mouth = classifier_face_and_mouth.w(1:end-1)'*(feats_mouth_and_face(:,all_test));
    
%     T1 = test_scores_region_all(iClass,all_test);
%     T1(isinf(T1)) = NaN;
%     T1(isnan(T1)) = min(T1(~isnan(T1)));
%     test_res_face_and_mouth = test_res_face_and_mouth + 0.1*T1;
    orig_test = ~[s40_fra.isTrain];
    orig_labels = [s40_fra.classID];
    orig_labels = 2*(orig_labels==curClass)-1;    
    orig_scores = -100*ones(size(s40_fra));
    orig_scores = orig_scores + rand(size(orig_scores ));
    %
    all_face_scores(iClass,:) = test_res_face;
    cur_face_det_scores = face_det_scores(all_test);
        
    
    aps_face = zeros(size(T_face));
    for ii = 1:length(T_face)
        D_face = cur_face_det_scores>T_face(ii);
%         D_face = top_face_scores(all_test)'>0;
        orig_scores([fra_db(all_test).imgIndex]) = test_res_face_and_mouth+1*double(D_face);
%     +.1*test_scores_region_all(iClass,orig_test);
        [~,~,f] = vl_pr(orig_labels(orig_test),orig_scores(orig_test));
        all_aps(iClass,ii) = f.ap;
    end
%     plot(T_face,all_aps(iClass,:))
    %
%     plot(T_face,aps_face)
%         
%     [perf_faces(iClass).recall, perf_faces(iClass).precision, perf_faces(iClass).info] = ...
%         vl_pr(orig_labels(orig_test),orig_scores(orig_test));
%     
%     [u,iu] = sort(orig_scores,'descend');
%     displayImageSeries(conf,fra_db(iu))
%     %
%     %      totalScores = test_res_face+.5*test_res_mouth;
    %    curClass = classes(iClass)
    %     totalScores = test_scores_region_all(iClass,:);
    
    %     vl_pr(orig_labels(orig_test),orig_scores(orig_test));
    %
    %     all_labels = 2*([fra_db(all_test).classID]==curClass)-1;
end
%%
plot(T_face,all_aps')
legend(conf.classes(classes));
%%
% create a classifier based on the entire dataset, for external usage

fra_classifiers = struct('class_name','classifier');
for iClass =1:4
    useTestExamples =true;
    curClass = classes(iClass);    
    toNormalize = false;
    bad_imgs =false(size(fra_db));
    fra_classifiers(iClass).class_name = classNames(iClass);
    fra_classifiers(iClass).classifier = train_classifier_helper(fra_db,feats_face,curClass,toNormalize,~bad_imgs,useTestExamples);% & (top_face_scores' > 0));
end


for iClass = 1:4
    fra_classifiers(iClass).classifier = fra_classifiers(iClass).classifier.w;
end
    
save ~/storage/misc/fra_classifiers.mat fra_classifiers


%% 
[u,iu] = sort(fra_classifiers(1).classifier.w(1:end-1)'*feats_mouth,'ascend');
displayImageSeries(conf,fra_db(iu),.1)


%% 
%

%
%     infos_ours = [perf_ours.info];
infos_ours_single_svm = [perf_faces.info];
%     ap_infos_ours = [infos_ours.ap]
ap_infos_ours_single_svm = [infos_ours_single_svm.ap]
infos_ours_tries{end+1} = ap_infos_ours;
%%
mixture_coeffs = 0:.01:.5;

combined_aps = zeros(length(classes),mixture_coeffs);

for ii = 1:length(mixture_coeffs)
    for iClass = 1:4
        curClass = classes(iClass);
        orig_test = ~[s40_fra.isTrain];
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

combined_aps(:,2)

