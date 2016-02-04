% clear classes;
initpath;
config;

% precompute the cluster responses for the entire training set.
%
conf.suffix = 'train_dt_noperson';
conf.VOCopts = VOCopts;
% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');
conf.detetion.params.detect_max_windows_per_exemplar = 1;
%%
baseSuffix = 'train_noperson_top_nosv';
conf.suffix = baseSuffix;

load dets_top_test;
load dets_top_train;

load top_20_1_train_patches

q_orig_t_d= train_patch_classifier(conf,[],[],'toSave',true,'suffix',...
    'retrain_consistent1_d','overRideSave',false);
[q_orig_t_d_test,q_orig_t_d_det] = applyToSet(conf,q_orig_t_d,test_ids,test_labels,...
    'ids_reorder_test_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);

[q_orig_t_d_train,q_orig_t_d_train_det] = applyToSet(conf,q_orig_t_d,train_ids,train_labels,...
    'ids_reorder_train_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);

% logreg!
for k = 1:length(q_orig_t_d_train)
    a = visualizeLocs2(conf,train_ids,q_orig_t_d_train(k).cluster_locs(1:100,:));
    figure(1),imshow(multiImage(a));title(num2str(k));
    pause;
end
%%
n = length(q_orig_t_d_train);
ws = zeros(1,n);
bs = zeros(1,n);

n_to_check =200;
for k = 1:n
    X  = q_orig_t_d_train(k).cluster_locs(1:n_to_check,12);
    y = train_labels(q_orig_t_d_train(k).cluster_locs(1:n_to_check,11));
    [ws(k),bs(k)] = logReg(X, y);
end

q_orig_t_d_test2 = applyLogReg(q_orig_t_d_test,ws,bs);
new_dets = det_union(q_orig_t_d_test2([1 5]));
[p,r,a,t,m] = calc_aps(new_dets,test_labels);
a
% plot(r,p)
% this beats state-of-the-art by 10% percent...
%
%% try to learn auxiliary parts for each cluster separately...
%% cans
r_1 = visualizeLocs2(conf,train_ids,q_orig_t_d_train(1).cluster_locs(1:30,:),...
    'inflateFactor',1,'height',128,'add_border',false);

figure,imshow(multiImage(r_1))
r_1_ids = r_1([2 3 7]);
r_1_samples = selectSamples(conf,r_1_ids);
r_ =rects2clusters(conf,r_1_samples,r_1_ids,[],0);
%r_1_cluster = makeCluster(cat(2,r_.cluster_samples),cat(1,r_.cluster_locs));
r_1_cluster = r_;
conf.clustering.num_hard_mining_iters = 10;
r_1_cluster = train_patch_classifier(conf,r_1_cluster,train_ids(~train_labels),...
    'suffix','r_1_clusts','overRideSave',false);

% detect the first in the others...

r_1_t = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs([2 3 7],:),...
    'inflateFactor',1.2,'height',128,'add_border',false);
conf.detection.params.detect_save_features = 1;
[r_1_cluster_det0] = applyToSet(conf,r_1_cluster(1),r_1_t,[],'r_1_cluster_det0','override',false);
conf.detection.params.detect_save_features = 0;
r_1_cluster1 = makeCluster(cat(2,r_1_cluster_det0.cluster_samples),...
    cat(1,r_1_cluster_det0.cluster_locs));
train_gt = train_labels(new_dets_train(1).cluster_locs(:,11));
conf.clustering.num_hard_mining_iters = 15;
r_1_cluster1_trained = train_patch_classifier(conf,r_1_cluster1,r_train(~train_gt),...
    'suffix','r_1_clusts1','overRideSave',false,'w1',1000);

[r_1_cluster1_res_san] = applyToSet(conf,r_1_cluster1_trained,r_train(train_gt),[],...
    'r_1_cluster1_res_san','nDetsPerCluster',50,'override',false,'disp_model',true);

[r_1_cluster1_res_train] = applyToSet(conf,r_1_cluster1_trained,r_train,[],...
    'r_1_cluster1_res_train','nDetsPerCluster',10,'override',false,'disp_model',true);

[r_1_cluster1_res_test] = applyToSet(conf,r_1_cluster1_trained,r_test,[],...
    'r_1_cluster1_res_test','nDetsPerCluster',10,'override',false,'disp_model',true);



%%
%% cans2
r_1_2 = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(1:30,:),...
    'inflateFactor',1,'height',128,'add_border',false);
figure,imshow(multiImage(r_1_2))
r_1_2_ids = r_1_2([2 3 7]);
r_1_2_samples = selectSamples(conf,r_1_2_ids);
r_ =rects2clusters(conf,r_1_2_samples,r_1_2_ids,[],1);
%r_1_2_cluster = makeCluster(cat(2,r_.cluster_samples),cat(1,r_.cluster_locs));
r_1_2_cluster = r_;
conf.clustering.num_hard_mining_iters = 10;
r_1_2_cluster = train_patch_classifier(conf,r_1_2_cluster,train_ids(~train_labels),...
    'suffix','r_1_2_clusts','overRideSave',true);

% detect the first in the others...

r_1_2_t = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs([2 3 7],:),...
    'inflateFactor',1.2,'height',128,'add_border',false);
conf.detection.params.detect_save_features = 1;
[r_1_2_cluster_det0] = applyToSet(conf,r_1_2_cluster(1),r_1_2_t,[],'r_1_2_cluster_det0','override',true);
conf.detection.params.detect_save_features = 0;
r_1_2_cluster1 = makeCluster(cat(2,r_1_2_cluster_det0.cluster_samples),...
    cat(1,r_1_2_cluster_det0.cluster_locs));
train_gt = train_labels(new_dets_train(1).cluster_locs(:,11));
conf.clustering.num_hard_mining_iters = 15;
r_1_2_cluster1_trained = train_patch_classifier(conf,r_1_2_cluster1,r_train(~train_gt),...
    'suffix','r_1_2_clusts1','overRideSave',true,'w1',1000);

[r_1_2_cluster1_res_san] = applyToSet(conf,r_1_2_cluster1_trained,r_train(train_gt),[],...
    'r_1_2_cluster1_res_san','nDetsPerCluster',50,'override',true,'disp_model',true);

[r_1_2_cluster1_res_train] = applyToSet(conf,r_1_2_cluster1_trained,r_train,[],...
    'r_1_2_cluster1_res_train','nDetsPerCluster',10,'override',true,'disp_model',true);

[r_1_2_cluster1_res_test] = applyToSet(conf,r_1_2_cluster1_trained,r_test,[],...
    'r_1_2_cluster1_res_test','nDetsPerCluster',10,'override',true,'disp_model',true);

visualizeLocs2(conf,r_test,r_1_2_cluster1_res_test.cluster_locs(1:20,:),'draw_rect',true);

%% hands...
r_2 = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(1:30,:),...
    'inflateFactor',1,'height',128,'add_border',false);
figure,imshow(multiImage(r_2))
r_2_ids = r_2([9 10]);
% conf_bu = conf;
r_2_samples = selectSamples(conf,r_2_ids);
r_ =rects2clusters(conf,r_2_samples,r_2_ids,[],1);
%r_2_cluster = makeCluster(cat(2,r_.cluster_samples),cat(1,r_.cluster_locs));
r_2_cluster = r_;
conf.clustering.num_hard_mining_iters = 10;
r_2_cluster = train_patch_classifier(conf,r_2_cluster,train_ids(~train_labels),...
    'suffix','r_2_clusts','overRideSave',true);

% detect the first in the others...

r_2_t = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(9:10,:),...
    'inflateFactor',1,'height',128,'add_border',false);
conf.detection.params.detect_save_features = 1;
[r_2_cluster_det0] = applyToSet(conf,r_2_cluster(1),r_2_t,[],'r_2_cluster_det0','override',false);
conf.detection.params.detect_save_features = 0;
r_2_cluster1 = makeCluster(cat(2,r_2_cluster_det0.cluster_samples),...
    cat(1,r_2_cluster_det0.cluster_locs));
train_gt = train_labels(new_dets_train(1).cluster_locs(:,11));
conf.clustering.num_hard_mining_iters = 15;
r_2_cluster1_trained = train_patch_classifier(conf,r_2_cluster1,r_train(~train_gt),...
    'suffix','r_2_clusts1','overRideSave',false,'w1',1000);

[r_2_cluster1_res_san] = applyToSet(conf,r_2_cluster1_trained,r_train(train_gt),[],...
    'r_2_cluster1_res_san','nDetsPerCluster',50,'override',false,'disp_model',true);

[r_2_cluster1_res_san1] = applyToSet(conf,r_2_cluster1_trained,r_train(686),[],...
    'r_2_cluster1_res_san1','nDetsPerCluster',50,'override',true,'disp_model',true);

visualizeLocs2(conf,r_train(686),r_2_cluster1_res_san1.cluster_locs,'draw_rect',true);

[r_2_cluster1_res_train] = applyToSet(conf,r_2_cluster1_trained,r_train,[],...
    'r_2_cluster1_res_train','nDetsPerCluster',10,'override',false,'disp_model',true);

[r_2_cluster1_res_test] = applyToSet(conf,r_2_cluster1_trained,r_test,[],...
    'r_2_cluster1_res_test','nDetsPerCluster',10,'override',false,'disp_model',true);

%% horz bottles...
r_3 = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(1:400,:),...
    'inflateFactor',1,'height',64,'add_border',false,'saveMemory',true);

r_3_= visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(gt_labels_train,:),...
    'inflateFactor',1,'height',64,'add_border',false,'saveMemory',true);

figure,imshow(multiImage(r_3_,find(gt_labels_train)))
r_3_ids = r_3([23]);
% conf_bu = conf;
conf = conf_bu;
r_3_samples = selectSamples(conf,r_3_ids);
r_ =rects2clusters(conf,r_3_samples,r_3_ids,[],1);
%r_3_cluster = makeCluster(cat(2,r_.cluster_samples),cat(1,r_.cluster_locs));
r_3_cluster = r_;
conf.clustering.num_hard_mining_iters = 10;
r_3_cluster = train_patch_classifier(conf,r_3_cluster,train_ids(~train_labels),...
    'suffix','r_3_clusts','overRideSave',true);

% detect the first in the others...

r_3_t = visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(23,:),...
    'inflateFactor',1,'height',128,'add_border',false);
conf.detection.params.detect_save_features = 1;
[r_3_cluster_det0] = applyToSet(conf,r_3_cluster(1),r_3_t,[],'r_3_cluster_det0','override',false);
conf.detection.params.detect_save_features = 0;
r_3_cluster1 = makeCluster(cat(2,r_3_cluster_det0.cluster_samples(:,1)),...
    cat(1,r_3_cluster_det0.cluster_locs(1,:)));
train_gt = train_labels(new_dets_train(1).cluster_locs(:,11));
conf.clustering.num_hard_mining_iters = 15;
r_3_cluster1_trained = train_patch_classifier(conf,r_3_cluster1,r_train(~train_gt),...
    'suffix','r_3_clusts1','overRideSave',false,'w1',1000);

[r_3_cluster1_res_san] = applyToSet(conf,r_3_cluster1_trained,r_train(train_gt),[],...
    'r_3_cluster1_res_san','nDetsPerCluster',50,'override',false,'disp_model',true);

[r_3_cluster1_res_train] = applyToSet(conf,r_3_cluster1_trained,r_train,[],...
    'r_3_cluster1_res_train','nDetsPerCluster',100,'override',false,'disp_model',true);

[r_3_cluster1_res_test] = applyToSet(conf,r_3_cluster1_trained,r_test,[],...
    'r_3_cluster1_res_test','nDetsPerCluster',100,'override',false,'disp_model',true);

%%
sel_ = 1;

r_clusters_train = [r_1_cluster1_res_train,r_2_cluster1_res_train,r_3_cluster1_res_train];
[M_train,gt_labels_train] = getAttributesForSVM(new_dets_train,r_clusters_train,train_labels);
M_train = M_train(:,sel_);
nTop =50;
% M_train = M
svmParams = '-t 0';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train,gt_labels_train,sum(train_labels));

r_clusters_test = [r_1_cluster1_res_test,r_2_cluster1_res_test,r_3_cluster1_res_test];

[M_test,gt_labels_test] = getAttributesForSVM(new_dets,r_clusters_test,test_labels);
M_test = M_test(:,sel_);
[p_l, ~, decision_values_test] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test,gt_labels_test,sum(test_labels));
a

%%

r = visualizeLocs2(conf,test_ids,new_dets.cluster_locs(1:100,:));

r = visualizeLocs2(conf,test_ids,new_dets.cluster_locs(t==1,:));

figure,imshow(multiImage(r,find(t==1)));
figure,plot(r,p);
title(num2str(a));

%% cups, bottles
q_orig_t_d_train2 = applyLogReg(q_orig_t_d_train,ws,bs);
new_dets_train = det_union(q_orig_t_d_train2);
r = visualizeLocs2(conf,train_ids,new_dets_train.cluster_locs(1:50,:),'add_border',false,...
    'height',round(64*1.2),'inflateFactor',1.2);
r = r([1:18 20]);
figure,imshow(multiImage(r));
% s = selectSamples(conf,r);
% s = s([1:18 20]);
% save new_rects s
load new_rects
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
newClusts2 = rects2clusters(conf2,s,r,[],1);
save newClusts2 newClusts2

r_train = ensureVisualization(conf2,new_dets_train,train_ids,'r_train_128.mat',1);

inflateFactor = 1;
a = visualizeLocs2(conf,train_ids,new_dets_train.cluster_locs(gt_labels_train,:),'saveMemory',...
    true,'add_border',false,'height',round(64*inflateFactor),'inflateFactor',inflateFactor,...
    'draw_rect',true);

r_test = ensureVisualization(conf2,new_dets,test_ids,'r_test_128.mat',1);

for k = 1:length(r_train)
    r_train{k} = imresize(r_train{k},1.5);
end
for k = 1:length(r_test)
    r_test{k} = imresize(r_test{k},1.5);
end

matlabpool
conf2.clustering.num_hard_mining_iters = 12;

newClusts2 = train_patch_classifier(conf2,newClusts2,r_train(~train_labels(new_dets_train.cluster_locs(:,11))),...
    'suffix','newClusts2','overRideSave',true);

t_ = cat(1,newClusts2.cluster_locs);
scales = t_(:,8);
figure,hist(scales);

conf2.detection.params.detect_levels_per_octave = 12;
conf2.detection.params.detect_max_scale = .8;
conf2.detection.params.detect_min_scale = .5;
conf2.detection.params.detect_levels_per_octave = 4;
[newClusts2_train] = applyToSet(conf2,newClusts2,r_train,[],...
    'newClusts2_train','nDetsPerCluster',10,'override',true,'disp_model',true);

t_ = cat(1,newClusts2_train.cluster_locs);
scales = t_(:,8);
figure,hist(scales);


[newClusts2_test] = applyToSet(conf2,newClusts2,r_test,[],...
    'newClusts2_test','nDetsPerCluster',10,'override',true,'disp_model',true);

train_labels_r = train_labels(new_dets_train.cluster_locs(:,11));
[A,AA] = visualizeClusters(conf2,r_train,newClusts2_train,'add_border',...
    true,'nDetsPerCluster',...
    10,'gt_labels',train_labels_r,...
    'disp_model',true);

% imwrite(clusters2Images(A),'newClusts2_train.jpg');

[M_train,gt_labels_train] = getAttributesForSVM(new_dets_train,newClusts2_train,train_labels);
%%
nTop =13;
svmParams = '-t 0';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train,gt_labels_train,sum(train_labels));
[M_test,gt_labels_test] = getAttributesForSVM(new_dets,newClusts2_test,test_labels);
[~, ~, decision_values_test] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test,gt_labels_test,sum(test_labels));
a

%%
plot(r,p,'-x')

%%

subDetectors = {};
for k = 1:5
    subDetectors{k} = makeSubDetector(conf,q_orig_t_d_train(k),q_orig_t_d_test(k),...
        ['orig' num2str(k)],'two_eyes',train_ids,test_ids,train_labels,test_labels);
end

%%
% apply each sub-detector to it's dataset...
for k = 1:length(subDetectors{k})
    mainDetTrainPatchesPath = fullfile(conf.cachedir,[['orig' num2str(k)] 'patches_train.mat']);
    mainDetTestPatchesPath = fullfile(conf.cachedir,[['orig' num2str(k)] 'patches_test.mat']);
    patches_main_train = ensureVisualization(conf,q_orig_t_d_train(k),train_ids,mainDetTrainPatchesPath);
    patches_main_test = ensureVisualization(conf,q_orig_t_d_test(k),test_ids,mainDetTestPatchesPath);
    conf2.detection.params.init_params.sbin = 4;
    [new_clusters_t,new_clusters_t_det,new_clusters_t_ap] = applyToSet(conf2,subDetectors{k},patches_main_train,[],...
        ['orig' num2str(k),'_two_eyes_train'],'nDetsPerCluster',10,'override',false,'disp_model',true);
    curTrainLabels = train_labels(q_orig_t_d_train(k).cluster_locs(:,11));
    curTestLabels = test_labels(q_orig_t_d_test(k).cluster_locs(:,11));
    [new_clusters_test,new_clusters_test_det,new_clusters_test_ap] = applyToSet(conf2,subDetectors{k},patches_main_test,[],...
        ['orig' num2str(k),'_two_eyes_test'],'nDetsPerCluster',10,'override',false,'disp_model',true);
    %     [a,aa] = visualizeClusters(conf2,patches_main_train,new_clusters_t(2),...
    %         'add_border',true,'gt_labels',curTrainLabels,'nDetsPerCluster',225);
    %     figure,imshow(multiImage(aa));
    
    [~,~,~,~,M_new_t] = calc_aps(new_clusters_t,curTrainLabels);
    [~,~,~,~,M_cur] = calc_aps(q_orig_t_d_train(k),train_labels);
    M_cur = M_cur(q_orig_t_d_train(k).cluster_locs(:,11),:);
    M_train = [M_new_t,M_cur];
    
    y_train = 2*double(train_labels(q_orig_t_d_train(k).cluster_locs(:,11)) > 0)-1;
    %%
    r_true = find(y_train==1);
    r_false = find(y_train==-1);
    r_true = r_true(1:5);
    y_train_ = y_train([r_true;r_false]);
    MM_train_ =M_train([r_true;r_false],:);
    model = svmtrain(y_train_, MM_train_,'-t 0');
    [predicted_label, accuracy, decision_values] = svmpredict(y_train,M_train,model);
    [p,r,a,t] = calc_aps2(decision_values,curTrainLabels,sum(train_labels));
    [~,~,~,~,M_new_test] = calc_aps(new_clusters_test,curTestLabels);
    [~,~,~,~,M_cur_test] = calc_aps(q_orig_t_d_test(k),test_labels);
    M_cur_test = M_cur_test(q_orig_t_d_test(k).cluster_locs(:,11),:);
    M_test = [M_new_test,M_cur_test];
    [predicted_label, accuracy, decision_values] = svmpredict(zeros(size(M_test(:,1))),...
        M_test,model);
    [prec,rec,aps,T,M] = calc_aps(q_orig_t_d_test(k),test_labels);
    curTestLabels = test_labels(q_orig_t_d_test(k).cluster_locs(:,11));
    [prec,rec,aps,T] = calc_aps2(decision_values,curTestLabels,sum(test_labels));
    aps
    %%
end

%%


for k = 1:5
    subDetectors{k} = makeSubDetector(conf,q_orig_t_d_train(k),...
        ['orig' num2str(k)],'two_eyes',train_ids,test_ids,train_labels,test_labels);
end

load top_20_1_train_patches
f=train_labels(dets_top_train(1).cluster_locs(:,11));
ff = find(f);
ff_neg = find(~f);
a_true= aa1(ff);
a_falses = aa1(ff_neg(1:20));
mouth_rects = selectSamples(conf,a_falses);
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
mouth_clusters = rects2clusters(conf2,mouth_rects,a_falses,[],1);
mouth_clusters= train_patch_classifier(conf2,mouth_clusters,a_true,'toSave',true,'suffix',...
    'mouth_clusters','overRideSave',false);
[mouth_clusters_t,mouth_det_t,mouth_ap_t] = applyToSet(conf2,mouth_clusters,aa1,[],...
    'mouth_clusters_t1','nDetsPerCluster',10,'override',false,'disp_model',true);
% [mouth_clusters_t2,mouth_det_t,mouth_ap_t] = applyToSet(conf2,mouth_clusters,aa1,[],...
%     'mouth_clusters_t2','nDetsPerCluster',10,'override',false,'disp_model',true,...
%     'dets',mouth_det_t,'useLocation',Z_mouth);
%
% Z_mouth = createConsistencyMaps(mouth_clusters_t,[64 64],ff_neg(1:20));
% for k = 1:length(Z_mouth)
%     Z_mouth{k} = imfilter(Z_mouth{k},fspecial('gauss',50,2));
%     Z_mouth{k} = (Z_mouth{k}/max(Z_mouth{k}(:))).^.25;
% end
% figure,imagesc(   Z_mouth{3} ); colorbar
%
% figure,imagesc(multiImage((Z_mouth(1:5)),true))

L =load('top_20_1_test_patches.mat');

% imshow(multiImage(L.aa1(1:25)))
matlabpool
nn = length(L.aa1);
[mouth_clusters_test,mouth_det_test,mouth_ap_t] = applyToSet(conf2,mouth_clusters,L.aa1,[],...
    'mouth_clusters_test','nDetsPerCluster',10,'override',false,'disp_model',true,'useLocation',0,...
    'dets',[]);

%%
%%train eyes...
imshow(multiImage(aa1(1:100),false,true))
choice_= [1:8];
imgChoice = aa1(choice_);
eyeRects = selectSamples(conf,imgChoice);
load eyeRects eyeRects
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
clusters = rects2clusters(conf2,eyeRects,imgChoice,[],1,0,false);
conf2_t = conf2;
top_labels = train_labels(dets_top_train(1).cluster_locs(:,11));
negatives = aa1(~top_labels);

nonPersonIds = getNonPersonIds(VOCopts);
conf2_t.max_image_size = 100;
clusts_eye = train_patch_classifier(conf2_t,clusters,nonPersonIds,'toSave',true,'suffix',...
    'clusts_eye','overRideSave',false);

[qq_eye,q_eye,aps_eye] = applyToSet(conf2,clusts_eye,aa1,[],'clusts_eye100','disp_model',true,...
    'add_border',false,'override',false,'dets',[],'useLocation',0);

[A,AA] = visualizeClusters(conf2,aa1,qq_eye(1),'add_border',...
    false,'nDetsPerCluster',100,'disp_model',true);

figure,imshow(multiImage(AA))

% combine the detection scores of eyes and mouths in the correct locations.
Z_eye = createConsistencyMaps(qq_eye,[64 64],1:10);
for k = 1:length(Z_eye)
    Z_eye{k} = imfilter(Z_eye{k},fspecial('gauss',21,3));
    Z_eye{k} = (Z_eye{k}/max(Z_eye{k}(:))).^.25;
end
figure,imagesc(  Z_eye{3} ); colorbar


[qq_eye_test,q_eye_test,aps_eye_test] = applyToSet(conf2,clusts_eye,L.aa1,[],'clusts_eye_test','disp_model',true,...
    'add_border',false,'override',false,'dets',[],'useLocation',false);

%% apply the mouth,eye clusters to the training set.
[prec_train,rec_train,aps_train,T_train,M_train] = calc_aps(dets_top_train(1),train_labels);
f_train = dets_top_train(1).cluster_locs(:,11);
M_train = M_train(f_train,:);
[prec_train_mouth,rec_train_mouth,aps_train_mouth,T_train_mouth,M_train_mouth] = ...
    calc_aps(mouth_clusters_t,train_labels(f_train));
[prec_train_eye,rec_train_eye,aps_train_eye,T_train_eye,M_train_eye] = ...
    calc_aps(qq_eye,train_labels(f_train));
MM_train = [M_train,M_train_mouth,M_train_eye];
y_train = 2*double(train_labels(f_train) > 0)-1;
M_train = [MM_train(:,1),max(M_train_mouth,[],2),max(M_train_eye,[],2)];

% model = svmtrain(y_train, M_train,'-t 1');


%%

[prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(dets_top_test(1),test_labels);
f_test = dets_top_test(1).cluster_locs(:,11);
M_test = M_test(f_test,:);
[prec_test_mouth,rec_test_mouth,aps_test_mouth,T_test_mouth,M_test_mouth] = ...
    calc_aps(mouth_clusters_test,test_labels(f_test));
[prec_test_eye,rec_test_eye,aps_test_eye,T_test_eye,M_test_eye] = ...
    calc_aps(qq_eye_test,test_labels(f_test));
MM = [M_test,M_test_mouth,M_test_eye];
%MM = MM(1:nn,:);
M_ = [MM(:,1),max(M_test_mouth,[],2),max(M_test_eye,[],2)];

%%
sel_ = 1:20:1000;
figure,imshow(multiImage(L.aa1(sel_),false,sel_));
%%
% imshow(multiImage(aa1(y_train>0),false,true))
% model = svmtrain(y(1:2:end), MM(1:2:end,:),'-t 2');

r_true = find(y_train==1);
r_false = find(y_train==-1);
r_true = r_true(1:3);

y_train_ = y_train([r_true;r_false]);
MM_train_ =MM_train([r_true;r_false],:);

model = svmtrain(y_train_, MM_train_,'-t 0');
% model.SVs'*model.sv_coef

[predicted_label, accuracy, decision_values] = svmpredict(zeros(size(MM,1),1),MM,model);
% decision_values(100:end) = -2;
t = -1.5;
% t2 = -2%-.75;
% regard the mouthness measure only if the face is detected with high
% confidence!
mouthness = (M_(:,2));
t2 = .5;
eyeness = M_(:,3);
% mouthness = t*mouthness.*(double(M_(:,1) > -.6) & eyeness>-.1);
M1 = M_(:,1)+mouthness+t2*eyeness;
% M1 = mouthness

M1 = decision_values;
% M1 = M_(:,2).*double(eyeness>-.1)+eyeness
% eyeness
% M_(:,2)

% plot(M1)

%  f = [single(M__) ;ones(1,size(M__,2), 'single')]'*[w;b];
% M1 = f;
ttt = test_labels(f_test);
% ttt(1:2:end) = 0;
[prec_test1,rec_test1,aps_test1,T_test] = calc_aps2(M1,ttt,sum(test_labels));

[s,is] = sort(M1,'descend');

% plot(cumsum(test_labels(f_test(is))))
%
s = M_(is,2);
s = round(s*10)/10;
aps_test1
%
% s=M_(is,1)
% s = round(s*100)/100;

%plot(rec_test1,prec_test1)
[r,ir] = sort(M1,'descend');
% [r,ir] = sort(M1.*test_labels(f_test),'descend');
% [r,ir] = sort(M1.*test_labels(f_test),'descend');
%

aa_ = L.aa1;
n_to_display = 20;
for k = 1:min(n_to_display,length(aa_))
    if (ttt(ir(k)))
        aa_{ir(k)} = imresize(addBorder(aa_{ir(k)},2,[0 255 0]),1);
    else
        aa_{ir(k)} = imresize(addBorder(aa_{ir(k)},2,[255 0 0]),1);
    end
end

figure,subplot(2,1,1),imshow(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display)))),.8));
% figure,imshow(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display))),false,true),.5));
subplot(2,1,2),bar(M_(ir(1:n_to_display),:));legend('drinking','mouthness','eyeness');
set(gca,'XTick',1:n_to_display);
% subplot(3,1,3),plot(rec_test1,prec_test1))
%
% figure,imshow(multiImage(L.aa1((ir(1:1:100)))))

%%
imwrite(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display))),false,s),.75),'nomouth.jpg')


%%

f = find(train_labels(q_orig_t_d_train2(1).cluster_locs(:,11)));

find(new_dets_train.cluster_locs(:,11)==853)

imshow(getImage(conf,train_ids{853}))

a1= visualizeLocs2(conf,train_ids,new_dets_train(1).cluster_locs(252,:),...
    'inflateFactor',1,'height',64,'add_border',false,'saveMemory',true,'draw_rect',true);
imshow(a1{1})

cur_ids = train_ids(q_orig_t_d_train2(1).cluster_locs(f,11));

r_tt= visualizeLocs2(conf,train_ids,q_orig_t_d_train2(1).cluster_locs(f,:),...
    'inflateFactor',1,'height',64,'add_border',false,'saveMemory',true);

figure,imshow(multiImage(r_tt))
% tt = [29 41 45 49];

tt = [65:70 72 76 78 80];

imshow(getImage(conf,cur_ids{tt(2)}))
% tt = 82;
tt = 1:length(cur_ids);
conf.detection.params.detect_max_windows_per_exemplar = 10;
[q_try,qq] = applyToSet(conf,q_orig_t_d_train2,cur_ids(tt),[],'try_false1',...
    'override',true,'dets',[]);

imshow(getImage(conf,train_ids{q_orig_t_d_train2(1).cluster_locs(f(25),11)}));

visualizeLocs2(conf,train_ids,q_orig_t_d_train2(1).cluster_locs(f(tt),:),...
    'inflateFactor',1,'height',128,'add_border',false,'draw_rect',true);


%% detect "cans" anywhere...

f_neg = find(~gt_labels_train);
model = trainDPM(conf,r_1(1:5),r_train(f_neg(1:100)),'with_cup');

cluster_locs = -10*ones(length(r_test),12);
for k = 1:1000%length(r_test)
    k
    [dets, boxes] = imgdetect(r_test{k}, model, model.thresh);
    boxes = reduceboxes(model, boxes);
    [dets boxes] = clipboxes(r_test{k}, dets, boxes);    
    I = nms(dets, 0.5);
    bb = dets(I,[1:4 end]);
    cluster_locs(k,1:4) = bb(1,1:4);
    cluster_locs(k,12) = bb(1,end);
    cluster_locs(k,11) = k;
end  

[p,ip] = sort(cluster_locs(:,12),'descend');


visualizemodel(model)
    

