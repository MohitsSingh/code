function full_results = run_pipeline(conf,fra_db,params,featureExtractor)
%RUN_PIPELINE Run full training/testing pipeline

full_results = [];
posClass = params.classes;
posClassName = params.posClassName;
commonOutDir = '~/storage/res_fra';
outDir = fullfile(commonOutDir,posClassName);
ensuredir(outDir);
[isClass,isValid,isTrain,f_train_pos,f_train_neg...
    f_test_pos,f_test_neg,f_train,f_test] = prepareMetadata(fra_db,posClass);
% start learning the different phases. For each phase, report both
% classification results using this phase alone and subsequent phase
% results.
% first extract global features (image,person,face) from all images
[global_feats_train,global_feats_test] = getGlobalFeatures_helper(commonOutDir);
% the pipeline....
cur_set = f_train;
samples = {};
% load ~/storage/misc/tmp.mat
params = setTestMode(params,false);
params.test_params = [];
params.debug_jump = 1;
%curPhases = params.phases(1);
% cur_set = f_train_pos(1:10:end);
phases = params.phases;

params.phases = params.phases(1);
params.logger.info('training first phase...','run_pipeline');
% cur_set = f_train_pos(1:10:end);
% cur_set = f_train_pos(2:10:end);
cur_set = f_train;

%classifier1Path = '~/storage/misc/classifier_phase_1.mat';
classifier1Path = '~/storage/misc/classifier_phase_1_interaction.mat';
if exist(classifier1Path,'file')
    load(classifier1Path);
else
    %     s1Path = '~/storage/misc/sampleData1.mat';
    s1Path = '~/storage/misc/sampleData1_interaction.mat';
    if exist(s1Path,'file')
        load(s1Path);
    else
        sampleData1 = collectSamples(conf, fra_db,cur_set,params);
        save(s1Path,'sampleData1','-v7.3');
    end
        
    cur_imgs = fra_db(cur_set);
    region_image_classes = [cur_imgs(sampleData1.inds).classID];
    [ii,jj] = find(isnan(sampleData1.feats))    
%     sel_ = region_image_classes == posClass;
%     sampleData1.inds = sampleData1.inds(sel_);
%     sampleData1.feats = sampleData1.feats(:,sel_);
%     sampleData1.labels = sampleData1.labels(sel_);
%     sampleData1.regions = sampleData1.regions(sel_);
%     sampleData1.ovps= sampleData1.ovps(sel_);
%     
    inds_folds = crossvalind_sub(sampleData1.inds,5);
    %inds_folds = crossvalind(method,length(labels),5);
    train_params = struct('classes',posClass,'toBalance',0,'lambdas',logspace(-4,-1,3));
    train_params.toBalance = -1;
    train_params.task = 'classification';
    
%     train_params.task = 'classification_rbf';            
    train_params.sample_from_positive_images_only = true;
    train_feats_normalized = normalize_vec(sampleData1.feats,1,1);
    [ii,jj] = find(isnan(train_feats_normalized))
    classifier1 = train_classifiers_folds(vl_homkermap(train_feats_normalized,1),...
        sampleData1.labels,sampleData1.ovps,train_params,inds_folds);
    save(classifier1Path,'classifier1','-v7.3');
end

params.phases(1).alg_phase.classifiers = classifier1;
% now use this to train the next stage
params.phases(2) = phases(2);
params = setTestMode(params,false)%,[true false]);

classifier2Path = '~/storage/misc/classifier_phase_2.mat';
if exist(classifier2Path,'file')
    load(classifier2Path);
else
    s2Path = '~/storage/misc/sampleData2.mat';
    if exist(s2Path,'file')
        load(s2Path);
    else
        sampleData2 = collectSamples(conf, fra_db,cur_set,params);
        save(s2Path,'sampleData2','-v7.3');
    end
    % [posFeats,negFeats] = splitFeats(samples,2*(labels==3)-1);
    train_params.task = 'classification';
    % classifiers1 = train_classifiers( sampleData1.feats,sampleData1.labels,sampleData1.ovps,train_params);
    train_params.toBalance = 0;
    train_params.lambdas=logspace(4,-1,3);
    
    %%%
    classifier2 = train_classifiers( sampleData2.feats,sampleData2.labels,sampleData2.ovps,train_params);
    save(classifier2Path,'classifier2','-v7.3');
end
%%%
params.phases(2).alg_phase.classifiers = classifier2;
params = setTestMode(params,true);
% params.phases(2).alg_phase.params.testMode = true;
% save classifiers.mat classifiers
% load classifiers.mat
% classifier = train_classifier_pegasos(feats,2*(labels==1)-1);
%%
% test_subset = f_train_pos(1:50:end);
test_subset = f_test(1:50:end);

% test_subset = f_test_pos(1:10:end);%:1:end);

params.test_params = [];
params.debug_jump = 1;
test_data_Path = '~/storage/misc/testData_interaction.mat';
if exist(test_data_Path,'file') && ~params.debug
    load(test_data_Path);
else
    testData = collectSamples2(conf, fra_db,test_subset,params);
    if ~params.debug
        save(test_data_Path,'testData','-v7.3');
    end
end

% save ~/storage/misc/tmp_test.mat test_feats test_labels test_regions test_inds -v7.3
params.classes = posClass;
class_names  = {};
% classifiers1 = struct;classifiers1.classifier_data = classifier;
%%
dlib_landmark_split;
%res = apply_classifiers(params.phases(2).alg_phase.classifiers,testData.feats,testData.labels,params);

test_feats_normalized = normalize_vec(testData.feats,1,1);
res = apply_classifiers(params.phases(1).alg_phase.classifiers,vl_homkermap(test_feats_normalized,1),testData.labels,params,false);

%%
% show some qualitative results.
[r,ir] = sort(res.curScores,'descend');
test_inds = testData.inds;
test_regions = testData.regions;
image_seen = false(size(fra_db));
for it = 1:length(r)
    t = ir(it);
    curInd = test_inds(t);
    if image_seen(curInd),continue,end
    image_seen(curInd) = true;
    imgData = fra_db(curInd);
    [I_sub,faceBox,mouthBox,I] = getSubImage2(conf,imgData,~params.testMode);
    %clf;imagesc2(I); plot_dlib_landmarks(imgData.Landmarks_dlib);    
    clf;imagesc2(I);
    plotPolygons(imgData.landmarks.xy,'g.','LineWidth',2);    
    %     dpc;continue
    f = test_inds==curInd;
    curRegions = test_regions(f);
    [mouthMask,curLandmarks] = getMouthMask(imgData,I_sub,mouthBox,imgData.isTrain);
    %     [mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
    heatMap = computeHeatMap_regions(I_sub,curRegions,res.curScores(f),'max');
    %     heatMap(heatMap<0) = min(heatMap(heatMap(:)>0));
    clf;
    curLandmarks = bsxfun(@minus,curLandmarks,mouthBox(1:2));
    subplot(1,3,1); imagesc2(I_sub);% plot_dlib_landmarks(curLandmarks);
    plotPolygons(curLandmarks,'go','MarkerSize',3,'LineWidth',3);
    subplot(1,3,2);
    imagesc2(sc(cat(3,heatMap,I_sub),'prob_jet'));
    % dpc;continue;
    subplot(1,3,3);
    curScores = res.curScores(f);
    displayRegions(I_sub,curRegions,curScores,'maxRegions',3);
end

function [global_feats_train,global_feats_test] = getGlobalFeatures_helper(commonOutDir);
globalFeatsPath = fullfile(commonOutDir,'global_feats.mat');
if (exist(globalFeatsPath,'file'))
    load(globalFeatsPath);
else
    global_feats_train = getGlobalFeatures(conf,fra_db(isTrain),featureExtractor);
    global_feats_test = getGlobalFeatures(conf,fra_db(~isTrain),featureExtractor);
    save(globalFeatsPath,'global_feats_train','global_feats_test');
end

function [isClass,isValid,isTrain,f_train_pos,f_train_neg...
    f_test_pos,f_test_neg,f_train,f_test] = prepareMetadata(fra_db,posClass)
isClass = [fra_db.classID] == posClass;
isValid = true(size(fra_db));%[fra_db.isValid];
isTrain = [fra_db.isTrain];
% findImageIndex(fra_db,'brushing_teeth_064.jpg')
train_pos = isClass & isTrain & isValid;
train_neg = ~isClass & isTrain & isValid;
f_train_pos = find(train_pos);
f_train_neg = find(train_neg);
test_pos = isClass & ~isTrain & isValid;
test_neg = ~isClass & ~isTrain & isValid;
f_test_pos = find(test_pos);
f_test_neg = find(test_neg);
f_train = find(isTrain & isValid);
f_test = find(~isTrain & isValid);


function params = setTestMode(params,testMode, testModes)
params.testMode = testMode;
if nargin < 3
    testModes = repmat(testMode,size(params.phases));
end
for i = 1:length(params.phases)
    params.phases(i).alg_phase.setTestMode(testModes(i));
end


