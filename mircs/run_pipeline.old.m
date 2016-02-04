function full_results = run_pipeline(conf,fra_db,params,posClass,posClassName,featureExtractor)
%RUN_PIPELINE Run full training/testing pipeline

commonOutDir = '~/storage/res_fra';
outDir = fullfile(commonOutDir,posClassName);
ensuredir(outDir);

% posClass = 4; % brushing teeth
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

phase_names = {'global','face','face_coarse','mouth'};
phases = struct('phase_name',{},'features',{},...
    'results',{});
params.isClass = isClass;

% start learning the different phases. For each phase, report both
% classification results using this phase alone and subsequent phase
% results.

% first extract global features (image,person) from all images
globalFeatsPath = fullfile(commonOutDir,'global_feats.mat');
if (exist(globalFeatsPath,'file'))
    load(globalFeatsPath);
else
    global_feats_train = getGlobalFeatures(conf,fra_db(isTrain),featureExtractor);
    global_feats_test = getGlobalFeatures(conf,fra_db(~isTrain),featureExtractor);
    save(globalFeatsPath,'global_feats_train','global_feats_test');
end

% get face detections + landmarks for all images

% % % train a global classifier
% % train_feats = [global_feats_train.global;global_feats_train.person];
% % train_labels = [fra_db(isTrain).classID];
% % train_params = struct('classes',posClass);
% % res_train_0 = train_classifiers(train_feats,train_labels,train_params,true);
% % 
% % test_feats = [global_feats_test.global;global_feats_test.person];
% % test_labels = [fra_db(~isTrain).classID];
% % 
% % class_names = {posClassName};
% % res = apply_classifiers(res_train_0,test_feats,test_labels,train_params,class_names);
% % plot(res.recall,res.precision);
% % [r,ir] = sort(res.curScores,'ascend');
% % ims = fra_db(~isTrain);ims = ims(ir);
% % displayImageSeries(conf,ims );
% for iPhase = 1:length(phase_names)
%     % collect data for phase
%     
%     % full mode: collect all feature from all stages
%     % train or test?

%     % --or -- %
%     % cascade mode: use previous stage to obtain 90% recall and continue
%     % from them. (not now)
%     
% end

% make a coarse region-of-interaction detector.
params_coarse = params;
params_coarse.cand_mode = 'boxes';
% params_coarse.nodes = nodes;

coarseDetectorPath = fullfile(outDir,'coarse_classifier.mat');
if (exist(coarseDetectorPath,'file'))
    load(coarseDetectorPath);
else
    [w_coarse b_coarse] = train_interaction_detector_coarse(conf,fra_db,f_train_pos,params_coarse,featureExtractor);
    save(coarseDetectorPath,'w_coarse','b_coarse');
end
params_coarse.w_int = w_coarse;
params_coarse.b_int = b_coarse;

% fra_db(1).valid = true;
% extract_all_features_lite(conf,fra_db(1),params);

statsPath = fullfile(outDir,'geomstats.mat');
if (exist(statsPath,'file'))
    load(statsPath);
else
    [~,sample_stats] = generateSamples(conf,fra_db(f_train_pos),true,params);
    save(statsPath,'sample_stats');
end
if 0
    % sample_stats = getSampleStats(pos_samples);
    neg_samples = generateSamples(conf,fra_db(f_train_neg),false,nodes,params,sample_stats);
    [feats,labels] = samplesToFeats([pos_samples,neg_samples],featureExtractor);            
    classifier_data = Pegasos(feats_n,labels(:),'lambda',.0001);%,...
    w = classifier_data.w(1:end-1);
    b = classifier_data.w(1:end-1);
    params.w = w;
    params.b = b;
end
%%

fineDetectorPath = fullfile(outDir,'fine_classifier_g.mat');
if (exist(fineDetectorPath,'file'))
    load(fineDetectorPath);
else
%     params.nodes = nodes;
    % params.isClass = isClass;
    params.phase = 'training';
    f_train_small = f_train;
    %[w_poi, b_poi, poi_train_samples] = poi_detector(conf,fra_db,f_train_small,params,params_coarse,featureExtractor);
    params.isClass = isClass;
    [w_poi, b_poi, poi_train_samples] = ...
        poi_detector(conf,fra_db,f_train_small,params,params_coarse,featureExtractor,sample_stats);    
    [feats,labels] = samplesToFeats(poi_train_samples,featureExtractor);
    feats = feats(1:4096,:);
    [normalizer,feats_n] = scaleFeatures(feats);
    classifier_data = Pegasos(feats,labels(:),'lambda',.0001);%,...
    w_poi = classifier_data.w(1:end-1);
    b_poi = classifier_data.w(end);
    save(fineDetectorPath,'w_poi','b_poi','feats','labels');
end

params.w_poi = w_poi;
params.b_poi = b_poi;
params.phase = 'testing';

f_test_small = f_test;
f_test_small = f_test_pos(8);
% f_test_small  = f_test(1:5:end);
%[~, ~, poi_test_results_small] = poi_detector(conf,fra_db,f_test_small(1:end),params,params_coarse,featureExtractor,sample_stats,normalizer);
[~, ~, poi_test_results_small] = poi_detector(conf,fra_db,f_test_small(1:end),params,params_coarse,featureExtractor,sample_stats);
pp = poi_test_results_small;

save poi_test_results_small1 poi_test_results_small
% 
% for t = 1:length(pp)
%     clc
%    t
%     clf; 
%     vl_tightsubplot(1,2,1);
%     imagesc2(pp(t).img);
%     vl_tightsubplot(1,2,2); 
%     p = pp(t);
%     q = displayRegions(p.img,p.mouthMask,[],'delay',0,'maxRegions',1,'dontPause',true,'show',false);
%     imagesc2(q);
%     pause
% end

%%
all_scores = -inf(10,length(pp));
for u = 1:length(pp)
    p = pp(u);
    z = p.scores;
    all_scores(1:length(z),u) = z;
end
coarse_scores = [pp.coarse_score];
all_scores = max(all_scores)+0*coarse_scores;
[v,iv] = sort(all_scores,'descend');
% iv = 1:length(iv);
% plot(all_scores)
labels =  isClass([pp.imgInd]);
% poi_detector(conf,fra_db,f_train_pos,params,params_coarse,featureExtractor,sample_stats);

%%
for it = 1:length(pp)
    t = iv(it)
    p = pp(t);
    
    if isClass(p.imgInd) || 1
        [z,iz] = sort(p.scores,'descend');
        maxRegions = 5;
        f = 1;
        mm = 2;
        nn = 3;
        clf; vl_tightsubplot(mm,nn,1);
        qs = {};
        ha = tight_subplot(mm,nn,[.1 .03],[.1 .1],[.01 .01]);
        axes(ha(1)); imagesc2(p.img);title('orig');axis off;
        for iq = 1:maxRegions
            q = displayRegions(p.img,p.masks(iq),p.scores(iq),'delay',0,'maxRegions',1,'dontPause',true,'show',false);
            axes(ha(iq+1)); imagesc2(q);title([num2str(iq) ': ' num2str(p.scores(iq))]);
            axis off;
            %             tight_subplot(mm,nn,iq+1);
            %             imagesc2(q);
        end
        
        %q = displayRegions(p.img,p.masks,p.scores,'delay',0,'maxRegions',3,'dontPause',false);
                dpc
%         saveas(gcf,sprintf('/home/amirro/notes/images/2015_07_30/false_%03.0f.png',it));
        %     imwrite(q,sprintf('/home/amirro/notes/images/2015_07_30/false_%03.0f.png',it));
    end
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
%
