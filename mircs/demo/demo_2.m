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
q_orig_t_d_test = applyToSet(conf,q_orig_t_d,test_ids,test_labels,...
    'ids_reorder_test_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);
q_orig_t_d_train = applyToSet(conf,q_orig_t_d,train_ids,train_labels,...
    'ids_reorder_train_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);
[prec,rec,aps,T,M]= calc_aps(q_orig_t_d_test,test_labels);
plot(rec,prec)
%%
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

save mouth_clusters mouth_clusters

[mouth_clusters_t,mouth_det_t,mouth_ap_t] = applyToSet(conf2,mouth_clusters,aa1,[],...
    'mouth_clusters_t1','nDetsPerCluster',10,'override',false,'disp_model',true);
[mouth_clusters_t2,mouth_det_t,mouth_ap_t] = applyToSet(conf2,mouth_clusters,aa1,[],...
    'mouth_clusters_t2','nDetsPerCluster',10,'override',false,'disp_model',true,...
    'dets',mouth_det_t,'useLocation',Z_mouth);

Z_mouth = createConsistencyMaps(mouth_clusters_t,[64 64],ff_neg(1:20));
for k = 1:length(Z_mouth)
    Z_mouth{k} = imfilter(Z_mouth{k},fspecial('gauss',50,2));
    Z_mouth{k} = (Z_mouth{k}/max(Z_mouth{k}(:))).^.25;
end
figure,imagesc(   Z_mouth{3} ); colorbar

figure,imagesc(multiImage((Z_mouth(1:5)),true))

L =load('top_20_1_test_patches.mat');

% imshow(multiImage(L.aa1(1:25)))
matlabpool
nn = length(L.aa1);
[mouth_clusters_test,mouth_det_test,mouth_ap_t] = applyToSet(conf2,mouth_clusters,L.aa1,[],...
    'mouth_clusters_test','nDetsPerCluster',10,'override',false,'disp_model',true,'useLocation',Z_mouth,...
    'dets',[]);

%%
%%train eyes...
imshow(multiImage(aa1(1:100)))
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
    'add_border',false,'override',false,'dets',[],'useLocation',Z_eye);

%% apply the mouth,eye clusters to the training set.

%%%
r_clusters_train = [qq_eye,mouth_clusters_t2];
[M_train,gt_labels_train] = getAttributesForSVM(dets_top_train(1),r_clusters_train,train_labels);
sel_ = 2:size(M_train,2);
M_train = M_train(:,sel_);
%%
nTop =1;
svmParams = '-t 2';
svmModel = trainAttributeSVM(M_train,gt_labels_train,nTop,svmParams);
[~, ~, decision_values_train] = svmpredict(zeros(size(M_train,1),1),M_train,svmModel);
[p,r,a,t] = calc_aps2(decision_values_train,gt_labels_train,sum(train_labels));
r_clusters_test = [qq_eye_test,mouth_clusters_test];
[M_test,gt_labels_test] = getAttributesForSVM(dets_top_test(1),r_clusters_test,test_labels);
M_test = M_test(:,sel_);
[p_l, ~, decision_values_test] = svmpredict(zeros(size(M_test,1),1),M_test,svmModel);
[p,r,a,t] = calc_aps2(decision_values_test,gt_labels_test,sum(test_labels));
a
%%
% good, now show the new detections re-sorted.....
% first , the original ones...
locs_orig = dets_top_test(1).cluster_locs;s
a_orig = visualizeLocs(conf,test_ids,locs_orig(1:20,:));
figure,imshow(multiImage(a_orig))
% now the next ones.
[~,is] = sort(decision_values_test,'descend');
locs_new = locs_orig(is,:);
locs_new(:,12) = decision_values_test(is);
a_new = visualizeLocs(conf,test_ids,locs_new(1:20,:));
figure,imshow(multiImage(a_new))

%%%
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

[predicted_label, accuracy, decision_values] = svmpredict(y_test,MM,model);
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
ttt = test_labels(f_test(1:nn));
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
aa_ = L.aa1(1:nn);
n_to_display = 20;
for k = 1:min(n_to_display,length(aa_))
    if (ttt(ir(k)))
        aa_{ir(k)} = imresize(addBorder(aa_{ir(k)},2,[0 255 0]),1);
    else
        aa_{ir(k)} = imresize(addBorder(aa_{ir(k)},2,[255 0 0]),1);
    end
end

figure,subplot(2,1,1),imshow(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display))),false,true),.8));
% figure,imshow(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display))),false,true),.5));
subplot(2,1,2),bar(M_(ir(1:n_to_display),:));legend('drinking','mouthness','eyeness');
set(gca,'XTick',1:n_to_display);
% subplot(3,1,3),plot(rec_test1,prec_test1))
% 
% figure,imshow(multiImage(L.aa1((ir(1:1:100)))))

%%
imwrite(imresize(multiImage(aa_(ir(1:min(length(aa_),n_to_display))),false,s),.75),'nomouth.jpg')