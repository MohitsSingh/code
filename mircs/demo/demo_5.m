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
    'retrain_consistent1_d','override',false);
% matlabpool
[q_orig_t_d_test,q_orig_t_d_det] = applyToSet(conf,q_orig_t_d,test_ids,test_labels,...
    'ids_reorder_test_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);

[q_orig_t_d_train,q_orig_t_d_train_det] = applyToSet(conf,q_orig_t_d,train_ids,train_labels,...
    'ids_reorder_train_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);

%imshow(multiImage(T_train.a(1:10:end),1:10:length(T_train.a)))
%figure,plot(q_orig_t_d_test(1).cluster_locs(:,12))
clocs_train = q_orig_t_d_train(1).cluster_locs;
face_thresh_train = clocs_train(281,12);
imshow(multiImage(T_train.a(clocs_train(:,12)>=face_thresh_train)));
clocs_test = q_orig_t_d_test(1).cluster_locs;
imshow(multiImage(T_test.a(clocs_test(:,12)>=face_thresh_train)));
%% create a cascade of some more detectors to improve the detection rate of the original one.
%% how good is the detector initially? 
[~,~,a] = calc_aps(q_orig_t_d_test,test_labels);
k = 1;% can/cup front view
curDetectorTrain = q_orig_t_d_train(k);
curDetectorTest = q_orig_t_d_test(k);
kk = 1;curSuffix = ['sub_' num2str(kk)];

subDetectors = {};
kk = 1;
[subDetectors{kk},conf2] = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);
kk = 2;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,false,[]);
kk = 3;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);

subDetectors = cat(2,subDetectors{:});

% check on a portion of the train set.
conf2.detection.params.detect_add_flip = 0;
T_train = load(fullfile(conf.cachedir,'orig1patches_train.mat'));
[subDetector_trainRes,subDet_resDet] = applyToSet(conf2,subDetectors,T_train.a,[],'subDetector_trainRes_n',...
    'override',false,'disp_model',true,'dets',[]);

kk = 1;curSuffix = ['sub_' num2str(kk)];
subDetectors = {};
[subDetectors{kk},conf2] = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);
subDetectors = cat(2,subDetectors{:});

[subDetector_trainRes,subDet_resDet] = applyToSet(conf2,subDetectors,T_train.a,[],'subDetector_trainRes_n1',...
    'override',true,'disp_model',true,'dets',[]);

m = visualizeLocs2(conf2,T_train.a,subDetector_trainRes(3).cluster_locs(1:10:500,:))
figure,imshow(multiImage(m))

%%
f = find(test_labels(curDetectorTest.cluster_locs(:,11)));
imwrite(multiImage(T_test.a(f),false),'T_test_true.jpg')

f = find(train_labels(curDetectorTrain.cluster_locs(:,11)));
imwrite(multiImage(T_train.a(f),false),'T_train_true.jpg')

%%
[p,r,a,t] = calc_aps(curDetectorTest,test_labels);
plot(r,p)

%%
Z = createConsistencyMaps(subDetector_trainRes,[64 64],[],10,[15 5])
for k = 1:length(Z)
    Z{k}= Z{k} > .5;
end
figure,imshow(multiImage(jettify(Z)));
% figure,imagesc(Z{5})
%%
% make the consistency maps only for mouths.
for k = [6:15]
    Z{k} = ones(size(Z{k}));
end
figure,imshow(multiImage(jettify(Z)))

[subDetector_trainRes] = applyToSet(conf2,subDetectors,T_train.a(1:end),[],'subDetector_trainRes_n',...
    'override',false,'disp_model',true,'dets',subDet_resDet,'useLocation',0,'uniqueImages',true);

% visualize the top 100 detections(original detector);
a0 = visualizeLocs2(conf,train_ids,curDetectorTrain.cluster_locs(1:100,:));
figure,imshow(multiImage(a0))

v = subDetector_trainRes(1).cluster_locs;
[g0,g1,g2] = intersect(1:100,v(:,11));
a1 = visualizeLocs2(conf2,T_train.a,v(1:100,:));
figure,imshow(multiImage(a1))
figure,imshow(T_train.a{v(91,11)})
figure,plot(v(1:100,8))
figure,plot(v(1:100,12))
[A,AA] = visualizeClusters(conf,T_train.a(1:end),subDetector_trainRes(1),'add_border',...
    false,'nDetsPerCluster',...
    500,'height',32,...
    'disp_model',false,'interactive',true);


imwrite(multiImage(AA_noloc(1:169),false),'noloc.tif');
imwrite(multiImage(AA(1:169),false),'loc.tif');

imwrite(clusters2Images(A([1:2 6:7 11:12])),'sub_detector.jpg')

T_test = load(fullfile(conf.cachedir,'orig1patches_test.mat'));

[subDetector_testRes] = applyToSet(conf2,subDetectors,T_test.a(1:end),[],'subDetector_testRes_n',...
    'override',false,'useLocation',0);

% two strategies; one is to throw everything in one large svm, the other is
% to try to train an "attribute" classifier, such as mouthness, eyeness,
% etc, by marking the amount of "mouthness" for each image. 

% try to find a good "eye" detector....
for q = 1:15
vis1 = visualizeLocs2(conf2,T_train.a, subDetector_trainRes(q).cluster_locs(1:3:300,:),'add_border',false);
figure(1),imshow(multiImage(vis1));title(num2str(q));
pause
end
% first, try a simple SVM, but remember to normalize the feature vector
% accordingly. 
%%
[M_train,gt_labels_train] = getAttributesForSVM(curDetectorTrain,[subDetector_trainRes],train_labels);
sel_ = 1:16;
M_train = M_train(:,sel_);
M_train(:,1) = M_train(:,1)*1;
sels = {1,2:5,6:11,12:16};
% M_train = consolidate(M_train,sels);
nTop =25;
% M_train = M
svmParams = '-t 0';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train,gt_labels_train,sum(train_labels));

[M_test,gt_labels_test] = getAttributesForSVM(curDetectorTest,subDetector_testRes,test_labels);
M_test = M_test(:,sel_);
% M_test = consolidate(M_test,sels);

[p_l, ~, decision_values_test] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test,gt_labels_test,sum(test_labels));
a

[s,is] = sort(decision_values_test,'descend');
%%
curLocsReordered1 = curDetectorTest.cluster_locs(is,:);
curLocsReordered1(:,12) = s;
a_res1 = visualizeLocs2(conf,test_ids,curLocsReordered1(1:100,:));
% good! gone from ~16 to ~22 ap with a single detector.
%% next! try number 5 - side view
k = 5;
kk = 1;
%% how good is the detector initially?
[~,~,a] = calc_aps(q_orig_t_d_test,test_labels);

k = 5;
curDetectorTrain = q_orig_t_d_train(k);
curDetectorTest = q_orig_t_d_test(k);
kk = 1;curSuffix = ['sub_' num2str(kk)];
subDetectors = {};
[subDetectors{kk},conf2] = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);
kk = 2;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,false,[]);
kk = 3;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);

subDetectors = cat(2,subDetectors{:});

% check on a portion of the train set.
T_train5 = load(fullfile(conf.cachedir,'orig5patches_train.mat'));
[subDetector5_trainRes] = applyToSet(conf2,subDetectors,T_train5.a(1:end),[],'subDetector5_trainRes',...
    'override',false);

T_train5 = load(fullfile(conf.cachedir,'orig5patches_train.mat'));
[subDetector5_trainRes] = applyToSet(conf2,subDetectors,T_train5.a(1:100),[],'subDetector5_trainCheck',...
    'override',true);

T_test5 = load(fullfile(conf.cachedir,'orig5patches_test.mat'));

[subDetector5_testRes] = applyToSet(conf2,subDetectors,T_test5.a(1:end),[],'subDetector5_testRes',...
    'override',false);

% two strategies; one is to throw everything in one large svm, the other is
% to try to train an "attribute" classifier, such as mouthness, eyeness,
% etc, by marking the amount of "mouthness" for each image. 

% first, try a simple SVM, but remember to normalize the feature vector
% accordingly. 

%%
[M_train,gt_labels_train] = getAttributesForSVM(curDetectorTrain,subDetector5_trainRes,train_labels);
sel_ = 1:size(M_train,2);
M_train = M_train(:,sel_);
M_train(:,1) = M_train(:,1)*1;

nTop =15;
svmParams = '-t 0';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train5] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train5,gt_labels_train,sum(train_labels));

[M_test,gt_labels_test] = getAttributesForSVM(curDetectorTest,subDetector5_testRes,test_labels);
M_test = M_test(:,sel_);
M_test(:,1) = M_test(:,1)*1;

[p_l, ~, decision_values_test5] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test5,gt_labels_test,sum(test_labels));
a
%%
[s,is] = sort(decision_values_test,'descend');
curLocsReordered5 = curDetectorTest.cluster_locs(is,:);
curLocsReordered5(:,12) = s;
a_res5 = visualizeLocs2(conf,test_ids,curLocsReordered5(1:100,:));
figure,imshow(multiImage(a_res5))
figure,imshow(multiImage(a_res1))

%%
%% 
%% next! try number 3 - straw
kk = 1;
%% how good is the detector initially?
[~,~,a] = calc_aps(q_orig_t_d_test,test_labels);

k = 3;
curDetectorTrain = q_orig_t_d_train(k);
curDetectorTest = q_orig_t_d_test(k);
kk = 1;curSuffix = ['sub_' num2str(kk)];
subDetectors = {};
[subDetectors{kk},conf2] = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);
kk = 2;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,false,[]);
kk = 3;curSuffix = ['sub_' num2str(kk)];
subDetectors{kk} = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);

subDetectors3 = cat(2,subDetectors{:});

% check on a portion of the train set.
T_train3 = load(fullfile(conf.cachedir,'orig3patches_train.mat'));
[subDetector3_trainRes] = applyToSet(conf2,subDetectors3,T_train3.a(1:end),[],'subDetector3_trainRes',...
    'override',false);

T_test3 = load(fullfile(conf.cachedir,'orig3patches_test.mat'));

[subDetector3_testRes] = applyToSet(conf2,subDetectors3,T_test3.a(1:end),[],'subDetector3_testRes',...
    'override',false);

% two strategies; one is to throw everything in one large svm, the other is
% to try to train an "attribute" classifier, such as mouthness, eyeness,
% etc, by marking the amount of "mouthness" for each image. 

% first, try a simple SVM, but remember to normalize the feature vector
% accordingly. 

[M_train,gt_labels_train] = getAttributesForSVM(curDetectorTrain,subDetector3_trainRes,train_labels);
sel_ = 1:size(M_train,2);
M_train = M_train(:,sel_);
M_train(:,1) = M_train(:,1)*1;

%%

nTop =15;
svmParams = '-t 3';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train3] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train3,gt_labels_train,sum(train_labels));

[M_test,gt_labels_test] = getAttributesForSVM(curDetectorTest,subDetector3_testRes,test_labels);
M_test = M_test(:,sel_);
M_test(:,1) = M_test(:,1)*1;

[p_l, ~, decision_values_test3] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test3,gt_labels_test,sum(test_labels));
a
%%
clusters_train_re = q_orig_t_d_train([1 3 5]);
clusters_train_re(1).cluster_locs(:,12) = decision_values_train;
clusters_train_re(2).cluster_locs(:,12) = decision_values_train3;
clusters_train_re(3).cluster_locs(:,12) = decision_values_train5;

clusters_test_re = q_orig_t_d_test([1 3 5]);

clusters_test_re(1).cluster_locs(:,12) = decision_values_test;
[s,is] = sort(clusters_test_re(1).cluster_locs(:,12),'descend');
clusters_test_re(1).cluster_locs = clusters_test_re(1).cluster_locs(is,:);
clusters_test_re(2).cluster_locs(:,12) = decision_values_test3;
[s,is] = sort(clusters_test_re(2).cluster_locs(:,12),'descend');
clusters_test_re(2).cluster_locs = clusters_test_re(2).cluster_locs(is,:);
clusters_test_re(3).cluster_locs(:,12) = decision_values_test5;
[s,is] = sort(clusters_test_re(3).cluster_locs(:,12),'descend');
clusters_test_re(3).cluster_locs= clusters_test_re(3).cluster_locs(is,:);

n_to_check =20
[ws,bs] = getLogRegCoefficients(clusters_train_re,train_labels,n_to_check);

clusters_test_re = applyLogReg(clusters_test_re,ws,bs);
clusters_united = det_union(clusters_test_re([1 3]));
[prec,rec,aps,T,M] = calc_aps(clusters_united,test_labels);
disp(aps)
% plot(rec,prec)
%%
a_res1_and_5 = visualizeLocs2(conf,test_ids,clusters_united.cluster_locs(1:50,:),'inflateFactor',1,...
    'height',64);
imshow(getImage(conf,test_ids{3125}))
getImageSet(conf,'train')

 figure,imshow(multiImage(a_res1_and_5(1:49)));
 imwrite(multiImage(a_res1_and_5(1:49)),'res.jpg','Quality',100)
 
figure,imshow(multiImage(a_res1_and_5(~test_labels(clusters_united.cluster_locs(1:200,11)))))
figure,imshow(multiImage(a_res1_and_5(test_labels(clusters_united.cluster_locs(:,11)))))


%% 
a_res1_and_5 = visualizeLocs2(conf,test_ids,clusters_united.cluster_locs(1:50,:),'inflateFactor',1,...
    'height',64);

figure,imshow(multiImage(a_res1_and_5))



%%
%% try to combine all of the decision values in a single svm.
t_ = inf;
% clusters_train_re = applyLogReg(clusters_train_re,ws,bs);
% clusters_test_re = applyLogReg(clusters_train_re,ws,bs);

X = inf(length(train_ids),2);
X(clusters_train_re(1).cluster_locs(:,11),1) = clusters_train_re(1).cluster_locs(:,12);
X(clusters_train_re(3).cluster_locs(:,11),2) = clusters_train_re(3).cluster_locs(:,12);
X(isinf(X(:))) = 0;
% xmin = min(X,[],1);
X(isinf(X(:,1)),1) = xmin(1);
X(isinf(X(:,2)),2) = xmin(2);
svmParams = '-t 2';
svmModel_c = trainAttributeSVM(X,train_labels,100,svmParams);

[~, ~, decision_values_train_c] = svmpredict(zeros(size(X,1),1),X,svmModel_c);
[p,r,a,t] = calc_aps2(decision_values_train_c,train_labels);
a
X_test = t_*ones(length(test_ids),2);
X_test(clusters_test_re(1).cluster_locs(:,11),1) = clusters_test_re(1).cluster_locs(:,12);
X_test(clusters_test_re(3).cluster_locs(:,11),2) = clusters_test_re(3).cluster_locs(:,12);
X_test(isinf(X_test(:))) = 0;
% xmin = min(X_test,[],1);
% X_test(isinf(X_test(:,1)),1) = xmin(1);
% X_test(isinf(X_test(:,2)),2) = xmin(2);

plot(X(~train_labels,1),X(~train_labels,2),'r+')
hold on;
plot(X(train_labels,1),X(train_labels,2),'g+')

[~, ~, decision_values_test_c] = svmpredict(zeros(size(X_test,1),1),X_test,svmModel_c);
[p,r,a,t] = calc_aps2(decision_values_test_c,test_labels);
a

%% do some visualization...
[M_test,gt_labels_test] = getAttributesForSVM(curDetectorTest,[subDetector_testRes],test_labels);
a = col(svmModel.SVs'*svmModel.sv_coef)';
% stem(a)
M_test_1_ = M_test;
%bsxfun(@times,M_test,a);
% figure,imagesc(M_train_1)
M_test_1 = consolidate(M_test_1_,sels);
bar(M_test_1(1:10,:))

legend({'drinking','eye','mouth','cup'})
[r,ir] = sort(decision_values_test,'descend');
n = 50;
aa = visualizeLocs2(conf,test_ids,curDetectorTest.cluster_locs(ir(1:n),:),'add_border',false);
figure,imshow(multiImage(aa));
% find the max. detection for this image.
% iir =20
%%
for iir = 1:n
    curRects = zeros(15,12);
    iir
for k = 1:length(subDetector_testRes)
    [a,b,c] = intersect(subDetector_testRes(k).cluster_locs(:,11),ir(iir));
    curRects(k,:) = subDetector_testRes(k).cluster_locs(b,:);    
end
    close all;
    [f1,if1] = max(curRects(1:5,12));
    [f2,if2] = max(curRects(6:10,12));
    [f3,if3] = max(curRects(11:15,12));
    
    f1 = M_test_1(ir(iir),2);
    f2 = M_test_1(ir(iir),3);
    f3 = M_test_1(ir(iir),4);
    
    h1 = subplot(1,2,1);imshow(aa{iir});
    hold on;
    plotBoxes2(curRects([if1+1],[2 1 4 3]),'r','LineWidth',2);
    plotBoxes2(curRects([if2+5],[2 1 4 3]),'--g','LineWidth',2);
    plotBoxes2(curRects([if3+10],[2 1 4 3]),'b','LineWidth',2);    
    h2 = subplot(1,2,2);bar(1,f1,'r');hold on;bar(2,f2,'g');hold on;bar(3,f3,'b');
    legend({'eye','mouth','cup'});
    set(gca,'ylim',[-3 3]);
%     axis equal
    
    set(gcf,'PaperPositionMode','manual');
    rect = get(gcf,'PaperPosition');
    rect = rect*.75;
    set(gcf,'PaperPosition',rect);
    set(gcf,'InvertHardCopy','on');
    set(gcf,'Renderer','painters');    
%         set(gca,'xticklabels',{'eye','mouth','cup'});

    saveas(gcf,sprintf('%03.0f.jpg',iir));
    pause;
    %print(gcf,sprintf('%03.0f.jpg',iir),'-dpsc');
    %print(gcf,sprintf('%03.0f.jpg',iir),'-dpsc');
end
%%
% h1Pos = get(h1,'Position');
% h2Pos = get(h2,'Position');
% h2Pos([2 4]) = .5*h1Pos([2 4]);
% set(h2,'Position',h2Pos);
%%
%% unite the detectors for training to find even more negatives patches...
[p,r,a,t] = calc_aps(clusters_train_re(1),train_labels);
a
plot(r,p)
% visualize
a_1_5_train = visualizeLocs2(conf,train_ids,clusters_train_re(1).cluster_locs(1:225,:),'add_border',false);

figure,imshow(multiImage(a_1_5_train(~train_labels(clusters_train_re(1).cluster_locs(1:50,11)))))
figure,imshow(multiImage(a_1_5_train(train_labels(clusters_train_re(1).cluster_locs(1:225,11)))))

a_1_5_train_false = a_1_5_train(~train_labels(clusters_train_re(1).cluster_locs(1:225,11)));
%sel_ = [2 4 7 8 10 11 13 18 72 75 159];
sel_ = 1:10;
a_1_5_train_false = a_1_5_train_false(sel_);

% rects = selectSamples(conf,a_1_5_train_false);
% save more_neg_mouths rects
load more_neg_mouths
clusters = rects2clusters(conf2,rects,a_1_5_train_false,[],1);
neg_mouths_t = train_patch_classifier(conf2,clusters,getNonPersonIds(VOCopts),...
    'suffix','neg_mouths_t','overRideSave',true);

% [neg_mouth_train5_res] = applyToSet(conf2,neg_mouths_t,T_train5.a,[],'neg_mouth_train5_res',...
%     'override',true);
% [neg_mouth_test5_res] = applyToSet(conf2,neg_mouths_t,T_test5.a,[],'neg_mouth_test5_res',...
%     'override',true);
% 
[neg_mouth_train_res] = applyToSet(conf2,neg_mouths_t,T_train.a,[],'neg_mouth_train_res',...
    'override',true);
[neg_mouth_test_res] = applyToSet(conf2,neg_mouths_t,T_test.a,[],'neg_mouth_test_res',...
    'override',true);

neg_mouth_u = getAttributesForSVM(q_orig_t_d_train(1),neg_mouth_train_res,train_labels);
neg_mouth_u5=getAttributesForSVM(q_orig_t_d_train(5),neg_mouth_train_res,train_labels);
%%
[~,~,a] = calc_aps(q_orig_t_d_test,test_labels);

k = 1;% can/cup front view
curDetectorTrain = q_orig_t_d_train(k);
curDetectorTest = q_orig_t_d_test(k);
kk = 1;curSuffix = ['sub_' num2str(kk) '_cup'];

subDetectors = {};
kk = 1;
[subDetectors{kk},conf2] = makeSubDetector(conf,curDetectorTrain,curDetectorTest,...
    ['orig' num2str(k)],curSuffix,train_ids,test_ids,train_labels,test_labels,true,[]);
subDetectors = subDetectors{1};

Z = createConsistencyMaps(subDetectors,[64 64],[],100,[100,9]);
figure,imshow(multiImage(jettify(Z)))

T_train = load(fullfile(conf.cachedir,'orig1patches_train.mat'));

f = find(train_labels(curDetectorTrain.cluster_locs(:,11)));
[subDetectors_cupTrain] = applyToSet(conf2,subDetectors,T_train.a(f),[],'subDetectors_cup',...
    'override',false,'disp_model',true,'useLocation',Z);

f_test = find(test_labels(curDetectorTest.cluster_locs(:,11)));
[subDetectors_cupTest,subDet_resDet] = applyToSet(conf2,subDetectors,T_test.a(f_test),[],'subDetectors_cupTest',...
    'override',false,'disp_model',true,'useLocation',0,'nDetsPerCluster',30);
%%

addpath(genpath('/home/amirro/code/3rdparty/berkeley_seg'));

%%
for k = 1:1:200
%     k = 11
r = (T_train.a{k});
r = imresize(r,1);

% run object

% imshow(r)
% gPb_orient = globalPb(r, '');
% 
% figure,imagesc(gPb_orient(:,:,3))

% figure,imagesc(r)
% figure,imagesc(max(gPb_orient,[],3))

E = edge(im2double(rgb2gray(r)),'canny');
figure(1),imshow([im2double(cat(3,E,E,E)),im2double(r)]);
pause;
end


T_train = visualizeLocs2(conf,train_ids,curDetectorTrain.cluster_locs(1:100,:),'add_border',false,...
'height',200,'inflateFactor',1.5);

mkdir('tmp1');
mkdir('tmp2');
curPath = pwd;
fPath = '/home/amirro/code/3rdparty/flandmark/bin/cpp/';
%%
f = find(train_labels);
for k = 11:1:20
    cd(curPath);
    r = (T_train{k});
    
    r = getImage(conf,train_ids{f(k)});
    
%     r = imresize(r,1);
    filePath =fullfile(curPath,'tmp1',[num2str(k) '.jpg']);
    outPath = fullfile(curPath,'tmp2',[num2str(k) '.jpg']);
    imwrite(r,filePath);
    cd(fPath);
    cmd = ['flandmark_1 ' filePath ' ' outPath];
    system(cmd);
end

%%
s = subDetector_trainRes(1);

A = visualizeLocs2(conf2,T_train.a,s.cluster_locs(1:1:100,:));
figure,imshow(multiImage(A));

inds1 = s.cluster_locs(:,11);
% train a detector using the rectangle from the second detection image....
c = 11;

% create virtual examples by scaling, rotating, etc this image slightly.

curTrainImage = T_train.a{inds1(5)};
imshow(curTrainImage)
curTrainImages = {};
theta = -5:1:5;
for itheta = 1:length(theta)
    I_rotated = imrotate(curTrainImage,theta(itheta),'bilinear','crop');
    curTrainImages{itheta} = I_rotated;
%     imshow(I_rotated);
%     pause;
end

% %cluster_locs = repmat(s.cluster_locs(c,:),length(curTrainImages),1);
%cluster_locs = repmat(s.cluster_locs(c,:),length(curTrainImages),1);
%conf,rects,pos_ids,inds,toShow,minOvp,crop_out,sameCluster
conf2.features.winsize = 4;
s2 = rects2clusters(conf2,s.cluster_locs(1:10,:), T_train.a(s.cluster_locs(1:10,11)),[],1,0,0,1);

% just make sure...
a = visualizeLocs2(conf2,T_train.a(s.cluster_locs(1:10,11)),s2.cluster_locs);
imshow(multiImage(a))

imshow(showHOG(conf,s2))

s2_trained = train_patch_classifier(conf2,s2,getNonPersonIds(VOCopts),'suffix','s2_trained','override',true);
imshow(showHOG(conf2,s2_trained))

% now run on the training set....
conf2.detection.params.detect_max_scale = .7;
[s2_resTrain] = applyToSet(conf2,s2_trained,T_train.a(1:end),[],'s2_resTrain',...
    'override',true,'disp_model',true,'dets',[]);

Z = createConsistencyMaps(s2_resTrain,[64 64],[],1000,[16 2])
figure,imagesc(Z{1})

v = 1:10:1000;
v(29)
figure,imshow(T_train.a{s2_resTrain.cluster_locs(281,11)})
a = visualizeLocs2(conf2,T_train.a(1:end),s2_resTrain.cluster_locs(1:10:1000,:));
figure,imshow(multiImage(a))

inds2 = s2_resTrain.cluster_locs(:,11);

k = 1:20:length(inds1);
n = length(inds1);
intersections = zeros(1,length(k));

for m = 1:length(k)
    %coeff = log(k(m))/log(nchoosek(n,k(m)));    
    coeff = log(nchoosek(n,k(m)));    
    intersections(m) = exp(log(length(intersect(inds1(1:k(m)),inds2(1:k(m)))))-coeff);
end

plot(k,intersections)



