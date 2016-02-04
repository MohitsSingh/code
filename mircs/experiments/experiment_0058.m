%% Experiment 0058: baseline model.
%% 29/1/2015
% 1. Extract fc6 features from stanford 40:
% images, person bounding boxes, faces, upper bodies
% 2. compute SVM to classify them
initpath;
config;
networkPath = 'imagenet-vgg-s.mat';
load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
s40_fra = s40_fra_faces_d;
load ~/storage/misc/all_dnn_feats.mat;
all_dnn_feats_deep = all_dnn_feats;
% load ~/storage/misc/all_dnn_feats_faces.mat
load ~/storage/misc/all_dnn_feats_head.mat
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
% all_dnn_feats_deep;

%%
all_dnn_feats = struct;
valids = true(size(s40_fra));
for u = 1:length(all_dnn_feats_deep)
    u
    all_dnn_feats(u).imageID = s40_fra(u).imageID;
    all_dnn_feats(u).feats_crop = cat(1,all_dnn_feats_deep(u).feats_crop(1).x); % should be crop
    all_dnn_feats(u).feats_crop_tiled = cat(1,col(all_dnn_feats_deep(u).feats_crop_tiled.x));
    all_dnn_feats(u).feats_full = cat(1,all_dnn_feats_deep(u).feats_full(1).x);
    all_dnn_feats(u).feats_full_tiled = cat(1,col(all_dnn_feats_deep(u).feats_full_tiled.x));
    all_dnn_feats(u).feats_crop_deep = cat(1,all_dnn_feats_deep(u).feats_crop_deep(1).x); % should be crop
    all_dnn_feats(u).feats_crop_deep_tiled = cat(1,col(all_dnn_feats_deep(u).feats_crop_deep_tiled.x));
    all_dnn_feats(u).feats_full_deep = cat(1,all_dnn_feats_deep(u).feats_full_deep(1).x);
    all_dnn_feats(u).feats_full_deep_tiled = cat(1,col(all_dnn_feats_deep(u).feats_full_deep_tiled.x));
    curHeadFeats = all_dnn_feats_head(u).result;
    if (isempty(curHeadFeats))
        valids(u) = false;
        continue;
    end
    for iType = 1:length(curHeadFeats)
        all_dnn_feats(u).(curHeadFeats(iType).type) = curHeadFeats(iType).feat;
    end
end
%%

extents = fieldnames(all_dnn_feats(1));
extents = setdiff(extents,{'imageID'});
% all_dnn_feats = rmfield(all_dnn_feats,'feats_deep');
[train_ids,~,train_labels] = getImageSet(conf,'train');
[test_ids,~,test_labels] = getImageSet(conf,'test');
% feat_layers = [16 16 18 18];
% feat_extent = {'feats_full','feats_crop','feats_full','feats_crop'};
% some training  parameters...
train_params.features.normalize_all = false;
train_params.features.normalize_each = false;
train_params.classes = [conf.class_enum.DRINKING];
% extract features from several image regions...
train_features = getImageFeatures(all_dnn_feats,train_ids);
% train_features is a cell array with several feature types.
%% test
test_features = getImageFeatures(all_dnn_feats,test_ids);

%% try all non-empty feature subsets (ablation study).

%%
% cool, now try to predict the usefullness of each features on its own.
% nTotalClasses = 40;
% classes = [conf.class_enum.DRINKING];%,conf.class_enum.SMOKING,conf.class_enum.BLOWING_BUBBLES,conf.class_enum.BRUSHING_TEETH];
% classes = my_classes
classes = 1:40;
nTotalClasses = length(classes);
avg_prec_est = zeros(nTotalClasses,length(train_features));
lambdas =  [1e-5 1e-6 1e-7];% 1e-6]
train_valids = valids(1:length(train_ids));
toBalance = 0;
for iClass = 1:nTotalClasses
    train_params.classes = classes(iClass);
    for iSubset = 19:length(train_features)
        feature_subset = iSubset
        train_features_1 = transform_features(train_features(feature_subset),train_params.features);
        res_train = train_classifiers(train_features_1(1:1:end,train_valids),train_labels(train_valids),train_params,toBalance,lambdas);
        avg_prec_est(iClass,iSubset) = res_train.classifier_data.optAvgPrec;
        clf; figure(1); imagesc(avg_prec_est(:,19:end)); drawnow
    end
end

%%
% now, train each class using several options:
% 1. each subset independently.
% 1. best classes by order (ir 1:9)

warning('using only deep features of entire image...');
test_results = struct('target_class',{},'feature_subset',{},'performance',{},'classifier_data',{});
for iClass = 1:nTotalClasses
    nExp = 0;
    train_params.classes = classes(iClass);
    % 1: single subsets
    %for iSubset = 1:length(train_features)
    
    for iSubset = 18
        fprintf('class: %d, subset: %d\n',train_params.classes, iSubset);
        nExp = nExp+1;
        feature_subset = iSubset;
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas);
    end
end
%

% 

for iClass = 1:nTotalClasses
    disp('***********')
    iClass
    nExp  = nRegions;
    train_params.classes = classes(iClass);
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    for nParams = 1:length(train_features) % note that we can skip the first one we don't to avoid later confusion
        nExp = nExp+1;
        feature_subset = ir(1:nParams);
        fprintf('class: %d, subset: %s\n',train_params.classes, num2str(feature_subset));
        lambdas = [1e-5];
        %         train_params.features.normalize_all = 0
        %         train_params.features.normalize_each = 1
        toBalance = 0
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas,toBalance);
        test_results(iClass,nExp).performance.info.ap
    end
end

%% check settings for different subsets, i.e, only non-deep, only deep features
classes = 1:40;
nTotalClasses = length(classes);

% start with very deep features only
is_deep = cellfun(@(x) any(strfind(x,'deep')),extents);
cur_sel = is_deep;
nSel = nnz(cur_sel);
for iClass = 1:nTotalClasses
    disp('***********')
    iClass
    nExp  = nRegions*2;
    train_params.classes = classes(iClass);
    cur_prec_est = avg_prec_est(iClass,:);
    cur_prec_est(~cur_sel) = -inf;
    [r,ir] = sort(cur_prec_est,'descend');
    ir = ir(1:n_deep);
    for nParams = 1:length(ir) % note that we can skip the first one but we don't to avoid later confusion
        nExp = nExp+1;
        feature_subset = ir(1:nParams);
        fprintf('class: %d, subset: %s\n',train_params.classes, num2str(feature_subset));
        lambdas = [1e-5];
        %         train_params.features.normalize_all = 0
        %         train_params.features.normalize_each = 1
        toBalance = 0
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas,toBalance);
        test_results(iClass,nExp).performance.info.ap
    end
end

%... and with non-deep features
cur_sel = ~is_deep;
nSel = nnz(cur_sel);
for iClass = 1:nTotalClasses
    disp('***********')
    iClass
    nExp  = nRegions*2.5;
    train_params.classes = classes(iClass);
    cur_prec_est = avg_prec_est(iClass,:);
    cur_prec_est(~cur_sel) = -inf;
    [r,ir] = sort(cur_prec_est,'descend');
    ir = ir(1:n_deep);
    for nParams = 1:length(ir) % note that we can skip the first one but we don't to avoid later confusion
        nExp = nExp+1;
        feature_subset = ir(1:nParams);
        fprintf('class: %d, subset: %s\n',train_params.classes, num2str(feature_subset));
        lambdas = [1e-5];
        %         train_params.features.normalize_all = 0
        %         train_params.features.normalize_each = 1
        toBalance = 0
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas,toBalance);
        test_results(iClass,nExp).performance.info.ap
    end
end

%aps = [test_results(9,:).performance];aps = [aps.info]; aps = [aps.ap];
% aps = [test_results.performance];aps = [aps.info]; aps = [aps.ap];

%%
save avg_prec_est2.mat test_results avg_prec_est
%%
% length(test_results);

% save avg_prec_est3_05.mat test_results

%%
% perfs = [test_results(:,1).performance];
% perfs = [perfs.info];
% perfs = [perfs.ap];

% stem(1:nTotalClasses,ap_6);
% view(90,90)
% set(gca,'XTickLabel',conf.classes)
% set(gca,'XTick',1:nTotalClasses)
% xlim([0 nTotalClasses+1]);
%%

% show the avg. rank of each extent

%avg_prec_est
%%
%view(90,90)
%%
my_classes = [conf.class_enum.DRINKING, conf.class_enum.SMOKING,conf.class_enum.BLOWING_BUBBLES,conf.class_enum.BRUSHING_TEETH,...
    conf.class_enum.LOOKING_THROUGH_A_MICROSCOPE,conf.class_enum.LOOKING_THROUGH_A_TELESCOPE]
mean(ap_6)
mean(ap_6(my_classes));
%%
% % % save all_test_results_2 avg_prec_est test_results
% save avg_prec_est.mat avg_pred_est
%%d
% test_results1 = reshape(test_results,[],nTotalClasses)';
for iClass =1:nTotalClasses
    perfs = [test_results(iClass,:).performance];
    perfs = [perfs.info]; perfs = [perfs.ap]
    clf,pdlot(1:9,perfs(10:end),'r-');title(conf.classes(iClass))
    ax = gca;
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    set(ax,'XTickLabel',extents(ir));
    %     clf,plot(1:9,perfs,'r-');title(conf.classes(iClass))
    pause
end
%%
% measure for each action class the maximal attained precision vs
% the precision using an increasing number of features, all features, etc.
close all
nRegions = length(extents);
nParams = 1:length(extents);
cumulativePerf = zeros(nRegions,nTotalClasses);
singlePerf = zeros(nRegions,nTotalClasses);
singlePerf_1 = zeros(nRegions,nTotalClasses);

for iClass = 1:nTotalClasses
    perfs = [test_results(iClass,:).performance];
    perfs = [perfs.info];
    [r,ir] = sort(avg_prec_est(iClass,:),'descend');
    perfs_1 = [perfs(1:nRegions).ap];
    perfs_2 = [perfs(nRegions+1:2*nRegions).ap]; % both
    %     perfs_2 = [perfs(2*nRegions+1:2.5*nRegions).ap]; perfs_1(1:2:end) = 0; % only deep
%     perfs_2 = [perfs(2.5*nRegions+1:end).ap]; perfs_1(2:2:end) = 0;  % non deep
    cumulativePerf(1:size(perfs_2,2),iClass) = perfs_2;
    singlePerf(:,iClass) = perfs_1(ir);
    singlePerf_1(:,iClass) = perfs_1;
end
cumulativePerf_orig = cumulativePerf;
ap_max = max(cumulativePerf_orig,[],1);
ap_max_single = max(singlePerf_1,[],1);

[m,im] = max(cumulativePerf,[],1);

% hist(im,1:nRegions)

% figure,plot(mean(cumulativePerf_orig'))

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

figure
ap_6 = cumulativePerf_orig(5,:);
barh(1:nTotalClasses,ap_6,.5,'hist')
ylim([0 nTotalClasses+1])
% ap_6(class_subset)
% conf.classes(class_subset)
grid on
set(gca,'YTickLabel',conf.classes)
set(gca,'YTick',1:nTotalClasses)

figure
p = [test_results(:,24).performance];
all_scores = cat(1,p.curScores);
all_labels = test_labels;
[~,idx_pred] = max(all_scores);
CM = confMatrix( test_labels, idx_pred, nTotalClasses );
imagesc(CM);
counts = sum(CM,2);
cm1 = bsxfun(@rdivide,CM,counts);
imagesc(cm1); colormap jet
mean(diag(cm1))
p = [p.info];[p.ap];
set(gca,'YTick',1:nTotalClasses)
set(gca,'YTickLabel',conf.classes)

fprintf('mean average precision (selected): %f\n',mean(ap_6));
fprintf('best average precision (oracle): %f\n',mean(ap_max));
fprintf('best average precision(best single per class): %f\n',mean(ap_max_single));
fprintf('best average precision(very deep features): %f\n',mean(singlePerf_1(16,:)));
fprintf('average precision(all features): %f\n',mean(cumulativePerf_orig(end,:)));

% fprintf('average precision(deep, full image): %f\n',mean(singlePerf(1,:)))
% fprintf('mean accuracy: %f\n',sum(diag(CM))/sum(CM(:)));


figure,
set(0,'defaulttextinterpreter','none')
title('single feature performance')
imagesc(singlePerf_1)
set(gca,'YTick',1:nRegions)
set(gca,'YTickLabel',extents)
set(gca,'XTick',1:nTotalClasses)
set(gca,'XTickLabel',conf.classes)
XYrotalabel(45,0)


%%
classes = [s40_fra.classID];
f = find(classes == conf.class_enum.LOOKING_THROUGH_A_MICROSCOPE);
conf.get_full_image = true

%% Collect for each image the locations of eye, mouth, etc.
bodyParts = collectBodyParts(conf,s40_fra);
%%
figure,imagesc(avg_prec_est)
[r,ir] = sort(avg_prec_est,2,'descend');
figure,imagesc(ir')
% set(gca,'YTick',1:nRegions)
% set(gca,'YTickLabel',extents)
% count for each feature how many times it appeared in each place.
q = zeros(nRegions);
for iClass = 1:nTotalClasses
    for n = 1:nRegions
        q(ir(iClass,n),n) = q(ir(iClass,n),n)+1;
    end
end
imagesc(q)
set(gca,'YTick',1:nRegions)
set(gca,'YTickLabel',extents)
set(gca,'XTick',1:nRegions)

z = repmat((1:nRegions),nRegions,1);
mean_ranking = sum(z.*q,2)./sum(q,2);
[u,iu] = sort(mean_ranking,'ascend');
extents(iu)
figure,imagesc(ir)

%% show the single feature performance...

%% use multi-class classification...
clf; figure(1)
train_features_1 = transform_features(train_features(16),train_params.features);
test_features_1 = transform_features(test_features(16),train_params.features);

addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');


curTrainLabels = train_labels(train_valids);
myStuff = 0;
if (myStuff)
    
    ii = 1:length(curTrainLabels);
    ii = randperm(length(curTrainLabels));
    curTrainLabels = curTrainLabels(ii);
else
    [curTrainLabels,ii] = sort(curTrainLabels,'ascend');
end

curTrainFeatures = train_features_1(:,train_valids);
toJump = 1;
curTrainFeatures = curTrainFeatures(1:toJump:end,ii);

[curTestLabels,ii_test] = sort(test_labels,'ascend');
curTestFeatures = test_features_1;
curTestFeatures = curTestFeatures(1:toJump:end,ii_test);

svm = train(curTrainLabels,...
    sparse(double(curTrainFeatures)),  ...
    sprintf(' -s 3 -B %f -c %f', ...
    1, .001), ...
    'col') ;

model.w = svm.w(:,1:end-1)' ;
model.b =  svm.w(:,end)' ;

scores = model.w' * curTestFeatures + model.b' * ones(1,size(curTestFeatures,2)) ;
[drop, imageEstClass] = max(scores, [], 1) ;
imageEstClass = svm.Label(imageEstClass);
% Compute the confusion matrix
% idx = sub2ind([length(classes), length(classes)], ...
%     test_labels', imageEstClass) ;
%
CM = confMatrix( curTestLabels, imageEstClass, nTotalClasses );
% CM = CM(svm.Label,:)

imagesc(CM);
counts = sum(CM,2);
cm1 = bsxfun(@rdivide,CM,counts);
imagesc(cm1); colormap jet
mean(diag(cm1))

%% sort by off-diagonal confusion, to show for each class the most confused one
[xx,yy] = meshgrid(1:size(cm1,1),1:size(cm1,1));
upper_mat = xx>yy;
off_diag = cm1.*upper_mat;
[m,im] = max(off_diag,[],2);
[q,iq] = sort(m,'descend');
z = [q,iq,im(iq)];
for n = 1:length(z)
    disp([num2str(z(n,1)) ', ' conf.classes{z(n,2)} ' --> ' conf.classes{z(n,3)}]);
end
