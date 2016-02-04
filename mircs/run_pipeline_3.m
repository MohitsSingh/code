function full_results = run_pipeline_3(conf,fra_db,params,featureExtractor)
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
phases = params.phases;
% params.phases = params.phases(1);
params.logger.info('training first phase...','run_pipeline');
cur_set = f_train;
% cur_set = f_train_pos;
% cur_set = cur_set(9:30);
classifiersPath = '~/storage/misc/classifiers.mat';

if exist(classifiersPath,'file')
    load(classifiersPath);
else    
    featuresPath = '~/storage/misc/features_int.mat';
    if exist(featuresPath,'file')
        load(featuresPath);
    else
        sampleData1 = collectSamples2(conf, fra_db,cur_set,params);
        save(featuresPath,'sampleData1','-v7.3');
    end        
    cur_imgs = fra_db(cur_set);
%     [ii,jj] = find(isnan(sampleData1.feats))          
    train_params = struct('classes',posClass,'toBalance',0,'lambdas',logspace(-5,0,5));
    train_params.toBalance =0;
    train_params.task = 'classification';   
    train_params.min_pos_ovp = .65;
    train_params.max_neg_ovp = .3;
    train_params
%     
%     train_params = rmfield(train_params,'min_pos_ovp');
%     train_params = rmfield(train_params,'max_neg_ovp');%     
    train_params.use_pos_images_only = true;
    train_params.hardnegative = false;
    %train_params.max_neg_ovp = .8;
%     train_feats = cat(2,sampleData1.feats{:,1});
%     train_feats = normalize_vec(train_feats,1,2);
%     train_feats = vl_homkermap(train_feats,1);
%     train_feats = train_feats;cat(2,sampleData1.
    [train_feats] = combineFeats(sampleData1.feats);
                %     
    classifier1 = train_classifiers(train_feats,sampleData1.labels,...
        sampleData1.ovps,train_params);
%     classifier2 = train_classifiers(train_feats1,sampleData1.labels,...
%         sampleData1.ovps,train_params);
    
    save(classifiersPath,'classifier1','-v7.3');
    
    %%    
    R = sampleData1.regions;
    showSorted(R,sampleData1.ovps);    
    x2(R(sampleData1.ovps>.8));        
    isPosInstance = sampleData1.labels==posClass & sampleData1.ovps>.7;    
    [r,ir] = sort(isPosInstance,'descend');
%     ir = randperm(size(f_,2));
    inds_s = sampleData1.inds_s;    
    got_images = zeros(1,size(f_,2));    
    for it = 1:size(f_,2)        
        k = ir(it);
        if r(it)>0,continue,end
        clf; 
        subplot(1,2,1);
        imagesc2(reshape(f_(:,k),9,9));
        title(num2str(sampleData1.labels(k)))
        subplot(1,2,2);        
        displayRegions(sampleData1.imgs{inds_s(k)},sampleData1.regions{k},[],'dontPause',true);
        if sampleData1.labels(k)==posClass
            got_images(inds_s(k)) = 1;
            sum(got_images)
        end
        dpc
    end
    % 
end
%%

% test_subset = f_train_pos(1:50:end);
% test_subset = f_test_pos(1:10:end);%:50:end);
test_subset = f_test;
params = setTestMode(params,true);
params.test_params = [];
params.debug_jump = 1;
test_data_Path = '~/storage/misc/testData.mat';
if exist(test_data_Path,'file') && ~params.debug
    load(test_data_Path);
else
    
%     for test_subset = f_test    
%         test_subset
        testData = collectSamples2(conf, fra_db,test_subset,params);
%     end
    getDataStats(conf,fra_db,test_subset,testData);
    if ~params.debug
        save(test_data_Path,'testData','-v7.3');
    end
end
params.classes = posClass;
class_names  = {};
% classifiers1 = struct;classifiers1.classifier_data = classifier;
%%
[train_feats] = combineFeats(sampleData1.feats);
                %     
classifier1 = train_classifiers(train_feats,sampleData1.labels,...
    sampleData1.ovps,train_params);
%%

dlib_landmark_split;
%res = apply_classifiers(params.phases(2).alg_phase.classifiers,testData.feats,testData.labels,params);
%test_feats = cat(2,testData.feats{:,1});
[test_feats] = combineFeats(testData.feats);
% test_feats = normalize_vec(test_feats,1,2);
%test_feats = vl_homkermap(test_feats,1);
res = apply_classifiers(classifier1,test_feats,testData.labels,params,false);
% res1 = apply_classifiers(classifier2,test_feats1,testData.labels,params,false);
res.info
% res1.info
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
%     if imgData.classID ~= posClass,continue,end
    debug_stuff = struct('calcFeats',true);
    %
%     T = collectSamples2(conf, fra_db,curInd,params,debug_stuff);        
%    displayRegions(T.imgs{1},T.regions,T.ovps);              
    [I_sub,faceBox,mouthBox,I] = getSubImage2(conf,imgData,~params.testMode);
    %clf;imagesc2(I); plot_dlib_landmarks(imgData.Landmarks_dlib);    
    clf;imagesc2(I);
    plotPolygons(imgData.landmarks.xy,'g.','LineWidth',2);    
    %     dpc;continue
    f = test_inds==curInd;
    curRegions = test_regions(f);
    curScores = res.curScores(f);
    displayRegions(I_sub,curRegions,curScores,'maxRegions',1);
    continue
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

