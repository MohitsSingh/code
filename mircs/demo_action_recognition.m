%% 8/2/2015
% Demo for Action Recognition using region selection

%%%%%
addpath('~/code/mircs');
initpath;
config;
conf.get_full_image = true;
roiParams = defaultROIParams();
landmarkParams = load('~/storage/misc/kp_pred_data.mat');
ptNames = landmarkParams.ptsData;
ptNames = {ptNames.pointNames};
requiredKeypoints = unique(cat(1,ptNames{:}));
landmarkParams.kdtree = vl_kdtreebuild(landmarkParams.XX,'Distance','L2');
landmarkParams.conf = conf;
landmarkParams.wSize = 96;
landmarkParams.extractHogsHelper = @(y) cellfun2(@(x) col(fhog2(im2single(imResample(x,[landmarkParams.wSize landmarkParams.wSize],'bilinear')))) , y);
landmarkParams.requiredKeypoints = requiredKeypoints;
landmarkInit = landmarkParams;
landmarkInit.debug_ = false;
nn_net = init_nn_network();
cd /home/amirro/code/3rdparty/voc-release5
startup
load ~/code/3rdparty/dpm_baseline.mat
initData = struct('landmarkParams',landmarkParams,'net',nn_net,'model',model);
%%%%%

imdb_type = 'stanford_40';
imdb_type = 'PASCAL2012';
imdb = getIMDB(imdb_type);

startup
params.cacheDir = '~/storage/voc_2012_data'
all_dnn_feats = struct;
for u = 1;length(imdb)
    extract_all_features_2(imdb(u),params,initData);
end

% images, person bounding boxes, faces, upper bodies
% 2. compute SVM to classify them
feat_matrix= {};
for u = 1:length(imdb)
    u
    curFeats =  extract_all_features_2(imdb(u),params,initData);
    feat_matrix{u} = curFeats;
end

%save(fullfile(params.cacheDir,'feats_matrix.mat'),'feat_matrix');

load (fullfile(params.cacheDir,'feats_matrix.mat'));

feat_matrix = cat(1,feat_matrix{:});
extents = {feat_matrix(1,:).type};

all_labels = [imdb.class_id];
train_inds = find([imdb.set]==1);
train_labels = all_labels(train_inds);
val_inds = find([imdb.set]==2); % val, actually
test_labels = all_labels(val_inds);

train_params.features.normalize_all = false;
train_params.features.normalize_each = true;
% extract features from several image regions...

train_features = getImageFeatures_2(feat_matrix(train_inds,:));
test_features = getImageFeatures_2(feat_matrix(val_inds,:));
classes = unique([imdb.class_id]);
train_params.classes=classes;

valids = true(size(imdb));
% train_features is a cell array with several feature types.
%% try all non-empty feature subsets (ablation study).
warning('currently trying only specific subset...');
% feature_subsets = allSubsets(length(train_features));
% feature_subsets = feature_subsets(2:end,:);
% feature_subsets = %[1 3 5 7]+1;
% feature_subsets  = [5 6 7 8 3 4 9];
% feature_subsets  = [5 6 8 2 3];
% feature_subsets  = 1:9
feature_subset  = [1 2 8 9];
% feature_subsets = [1:8];
%feature_subsets = [1 0 1 0]>0;
% feature_subset = ir(1:5)
% feature_subsets = 5:9
% feature_subsets = 7
train_params.features.normalize_each =0;
train_params.features.normalize_all = 0;
train_features_1 = transform_features(train_features(feature_subset),train_params.features);
res_train = train_classifiers(train_features_1,train_labels,train_params,valids(1:length(train_inds)));
%% test
test_features_1 = transform_features(test_features(feature_subset),train_params.features);
res_test = apply_classifiers(res_train,test_features_1,test_labels,train_params,VOCopts.actions)
res_test.info.ap
% summarsizeResults(res_test);
%load ~/storage/mircs_18_11_2014/s40_fra.
% deep_features = get_deep_features();
% params.feat_type = 'fc6';
%%

% cool, now try to predict the usefullness of each features on its own.
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
nTotalClasses = length(classes);
avg_prec_est = zeros(nTotalClasses,length(train_features));
for iClass = 1:nTotalClasses
    train_params.classes = classes(iClass);
    for iSubset = 1:length(train_features)
        feature_subset = iSubset
        train_features_1 = transform_features(train_features(feature_subset),train_params.features);
        res_train = train_classifiers(train_features_1(1:1:end,:),train_labels,train_params,valids(1:length(train_inds)));
        avg_prec_est(iClass,iSubset) = res_train.classifier_data.optAvgPrec;
                
    end
end

% compute mutual information between feats and labels...
feature_subset = 1:9;
train_features_1 = transform_features(train_features(feature_subset),train_params.features);
means = mean(train_features_1,2);
train_features_1_1 = bsxfun(@gt,train_features_1,means);
bads = sum(abs(train_features_1),2) < 50;
nTotalClasses = length(classes);
mi_est = zeros(nTotalClasses,size(train_features_1,1));
for iClass = 1:nTotalClasses
    train_params.classes = classes(iClass);
    %     for iSubset = 1:length(train_features)
    cur_labels = train_labels==classes(iClass);
%     figure,plot(sum(abs(train_features_1),2))
    for p = 1:size(train_features_1,1)
        if (mod(p,500)==0)
            p
        end
        if (~bads(p))
            mi_est(iClass,p) = MI(cur_labels, train_features_1_1(p,:));
        end
    end            
%     res_train = train_classifiers(train_features_1(1:1:end,:),train_labels,train_params,valids(1:length(train_inds)));
%     mi_est(iClass,iSubset) = res_train.classifier_data.optAvgPrec;    
    %     end
end

% now, train each class using several options:
% 1. each subset independently.
% 1. best classes by order (ir 1:9)
test_results = struct('target_class',{},'feature_subset',{},'performance',{},'classifier_data',{});
for iClass = 1:nTotalClasses
    nExp = 0;
    train_params.classes = classes(iClass);
    % 1: single subsets
    for iSubset = 1:length(train_features)
        nExp = nExp+1;
        feature_subset = iSubset;
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_inds,test_features);
    end
    
    %     perfs = [test_results.performance];
    %     perfs = [perfs.info]; perfs = [perfs.ap];
    %     figure,plot(avg_precs_lite(iClass,:),perfs,'r+')
    %
    % 2: sorted subsets
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    for nParams = 1:length(train_features) % note that we can skip the first one we don't to avoid later confusion
        nExp = nExp+1;
        feature_subset = ir(1:nParams);
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_inds,test_features);
    end
    %
    %         perfs = [test_results.performance];
    %     perfs = [perfs.info]; perfs = [perfs.ap];
    %     figure,plot(1:18, perfs,'r+')
end

save all_test_results_pascal avg_prec_est test_results

% save mi_est mi_est means

%%
% test_results1 = reshape(test_results,[],nTotalClasses)';
for iClass =1:nTotalClasses
    perfs = [test_results(iClass,:).performance];
    perfs = [perfs.info]; perfs = [perfs.ap];
    clf,plot(1:9,perfs(10:18),'r-');title(conf.classes(iClass))
    ax = gca;
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    set(ax,'XTickLabel',extents(ir));
    %     clf,plot(1:9,perfs,'r-');title(conf.classes(iClass))
    pause
end
%%
% measure for each action class the maximal attained precision vs
% the precision using an increasing number of features, all features, etc.

cumulativePerf = zeros(9,nTotalClasses);
singlePerf = zeros(9,nTotalClasses);
for iClass =1:nTotalClasses
    perfs = [test_results(iClass,:).performance];
    perfs = [perfs.info];
    cumulativePerf(:,iClass) = [perfs(10:18).ap];
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    singlePerf(:,iClass) = [perfs(ir).ap];
end
close all
ap_6 = cumulativePerf(6,:);
ap_max = max(cumulativePerf,[],1);
[m,im] = max(cumulativePerf,[],1);

singlePerf = bsxfun(@rdivide,singlePerf,max(cumulativePerf));
cumulativePerf = bsxfun(@rdivide,cumulativePerf,max(cumulativePerf));
% figure,plot(cumulativePerf)

mean_perf = mean(cumulativePerf,2);

figure(2); clf;
errorbar(mean(cumulativePerf,2),std(cumulativePerf'));
hold on
hold on; plot(median(cumulativePerf,2),'g-')
plot(mean(singlePerf,2),'r-d');
legend({'cumulative performance mean','cumulative performance median','single feature performance'})
ylabel('avg. normalized class performance');
xlabel('no. regions selected');
grid on
% print the sorted extent for a few classes...
fid = fopen('~/notes/region_selection_pascal.txt','w');
for iClass = 1:length(classes)
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    fprintf(fid,'%s (%02.2f):', VOCopts.actions, 100*ap_6(iClass));
    for u = 1:5
        fprintf(fid,'%s,',extents{ir(u)});
    end
    fprintf(fid,'\n');
end
fclose(fid);
% mean_ap

%     extents(ir)
% end
% z = zeros(nTotalClasses,length();

% [r,ir] = sort(avg_precs_lite,'descend')
% extents = {train_features.extent};
% extents(ir)

%% test if the mean avg. precision on validation indeed means something...
%% try all non-empty feature subsets (ablation study).
aps = zeros(size(avg_prec_est));
avg_precs_2 = zeros(size(avg_prec_est));
warning('currently trying only specific subset...');
for iSubset = 1:length(avg_prec_est)
    feature_subset  = ir(1:4)
    train_features_1 = transform_features(train_features(feature_subset),train_params.features);
    res_train = train_classifiers(train_features_1,train_labels,train_params,valids(1:length(train_inds)));
    avg_precs_2(iSubset) = res_train.classifier_data.optAvgPrec;
    % test
    test_features = getImageFeatures(all_dnn_feats,val_inds);
    test_features_1 = transform_features(test_features(feature_subset),train_params.features);
    res_test = apply_classifiers(res_train,test_features_1,test_labels,train_params);
    aps(iSubset) = res_test.info.ap
end


%% check the same stuff but using the mutual information
test_results = struct('target_class',{},'feature_subset',{},'performance',{},'classifier_data',{});
train_features_1 = transform_features(train_features,train_params.features);
[q,iq] = sort(mi_est,2,'descend');
test_features_1_1 = bsxfun(@gt,transform_features(test_features,train_params.features),means);
%%
aps = zeros(nTotalClasses,1);
for iClass = 1:nTotalClasses
    nExp = 0;
    train_params.classes = classes(iClass);
    % 1: single subsets
    %     for iSubset = 1:length(train_features)
    nExp = nExp+1;
    mi_feature_subset = iq(iClass,1:2000);   
    lambdas = logspace(-4,-1,3);    
    res_train = train_classifiers(2*single(train_features_1_1(mi_feature_subset,:))-1,train_labels,train_params,valids(1:length(train_inds)),lambdas);   
    res_test = apply_classifiers(res_train,2*single(test_features_1_1(mi_feature_subset,:))-1,test_labels,train_params);
    aps(iClass) = res_test.info.ap;
%     test_results.target_class = iClass;
%     test_results.feature_subset = feature_subset;
%     test_results.performance = res_test;
%     test_results.classifier_data = res_train.classifier_data;
    %     rend
end
