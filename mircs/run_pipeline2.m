function full_results = run_pipeline(conf,fra_db,params,posClass,posClassName,featureExtractor)
%RUN_PIPELINE Run full training/testing pipeline

full_results = [];

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
debug_jump = 1;
load ~/storage/misc/tmp.mat
params.testMode = false;
params.test_params = [];
% [feats,labels,regions,inds] = collectSamples(conf, fra_db,cur_set,params,featureExtractor,debug_jump);
%save ~/storage/misc/tmp.mat feats labels regions inds
% [posFeats,negFeats] = splitFeats(samples,2*(labels==3)-1);
train_params = struct('classes',1,'toBalance',0,'lambdas',logspace(-4,-1,3));
classifiers = train_classifiers( feats,labels,train_params);
% classifier = train_classifier_pegasos(feats,2*(labels==1)-1);
debug_jump_test = 1;


test_subset = 1;
f_test = f_test(test_subset);
params.testMode = true;
params.test_params = [];

[test_feats,test_labels,test_regions,test_inds] = collectSamples(conf, fra_db,f_test,params,featureExtractor,debug_jump_test);
% save ~/storage/misc/tmp_test.mat test_feats test_labels test_regions test_inds -v7.3
params.classes = 1;
class_names  = {};
% classifiers1 = struct;classifiers1.classifier_data = classifier;
res = apply_classifiers(classifiers,test_feats,test_labels,params,class_names)
% show some qualitative results.
[r,ir] = sort(res.curScores,'descend');
image_seen = false(size(fra_db));
for it = 1:length(r)
    t = ir(it);
    curInd = test_inds(t);
    if image_seen(curInd),continue,end
    image_seen(curInd) = true;
    [I_sub,faceBox,mouthBox] = getSubImage2(conf,fra_db(curInd));
    f = test_inds==curInd;
    curRegions = test_regions(f);
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


function globalFeats = getGlobalFeatures(conf,fra_db,featureExtractor)
globalFeats = struct('global',{},'person',{});
I_full = {};
I_person = {};
I_face = {};
tic_id = ticStatus('extracting global features...',.5,.5);
for iImg = 1:length(fra_db)
    [I,I_rect] = getImage(conf,fra_db(iImg));
    I_full{iImg} = I;
    I_person{iImg} = cropper(I,I_rect);
    I_face{iImg} = cropper(I,round(fra_db(iImg).faceBox));
    tocStatus(tic_id,iImg/length(fra_db));
end

globalFeats(1).global = featureExtractor.extractFeaturesMulti(I_full);
globalFeats(1).person = featureExtractor.extractFeaturesMulti(I_person);
globalFeats(1).face = featureExtractor.extractFeaturesMulti(I_face);

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
