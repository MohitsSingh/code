initpath;
config;

% precompute the cluster responses for the entire training set.
%
conf.suffix = 'train_dt_noperson';
conf.VOCopts = VOCopts;
% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);

conf.detetion.params.detect_max_windows_per_exemplar = 1;

load dets_top_train
load dets_top_test
%%

load top_20_1_train_patches

matlabpool

%% it seems the consistency reorders the true detections nicely. now I have to learn
%% a good classifier for such detections...

conf.detection.params.detect_save_features = 1;
q_orig = applyToSet(conf,cur_top_dets,ids_orig,[],'ids_orig_check','override',false,...
    'nDetsPerCluster',inf);
conf.detection.params.detect_save_features = 0;
% q_orig;
[train_ids,train_labels] = getImageSet(conf,'train');
[test_ids,test_labels] = getImageSet(conf,'test');
q_orig_t_d= train_patch_classifier(conf,[],[],'toSave',true,'suffix',...
    'retrain_consistent1_d','overRideSave',false);
q_orig_t_d_train = applyToSet(conf,q_orig_t_d,train_ids,train_labels,...
    'ids_reorder_train_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);
q_orig_t_d_test = applyToSet(conf,q_orig_t_d,test_ids,test_labels,...
    'ids_reorder_test_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true)

load a_train_inf;
load a_test_inf

gt_labels = train_labels(q_orig_t_d_train(1).cluster_locs(:,11));
plot(gt_labels)

%% 
%%train eyes...
imshow(multiImage(aa1(1:100),false,true))
choice_= [1:8];
imgChoice = aa1(choice_);
eyeRects = selectSamples(conf,imgChoice);
save eyeRects eyeRects
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
clusters = rects2clusters(conf2,eyeRects,imgChoice,[],1,0,false);
conf2_t = conf2;
top_labels = train_labels(dets_top_train(1).cluster_locs(:,11));
negatives = aa1(~top_labels);

nonPersonIds = getNonPersonIds(VOCopts);
conf2_t.max_image_size = 100;
clusts_eye = train_patch_classifier(conf2_t,clusters,nonPersonIds,'toSave',true,'suffix',...
    'clusts_eye','overRideSave',true);

[qq_eye,q_eye,aps_eye] = applyToSet(conf2,clusts_eye,aa1,[],'clusts_eye100','disp_model',true,...
    'add_border',false,'override',false,'dets',q_eye,'useLocation',0);

[A,AA] = visualizeClusters(conf2,aa1,qq_eye(1),'add_border',...
      false,'nDetsPerCluster',100,'disp_model',true);
  
figure,imshow(multiImage(AA))

%%
% add some more mouth forms....
aa1
imshow(multiImage(aa1(1:10:1000)))
% detect mouths with toothbrush, bubbles
t_f = find(~top_labels);
imshow(multiImage(aa1(t_f(1:100)),false,true))
choice_=[1:20]
imgChoice = aa1(t_f(choice_));
mouthRects_b= selectSamples(conf,imgChoice);
save mouthRects_b mouthRects_b
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
clusters = rects2clusters(conf2,mouthRects_b,imgChoice,[],1,0,false);
conf2_t = conf2;
top_labels = train_labels(dets_top_train(1).cluster_locs(:,11));
negatives = aa1(top_labels);

conf2_t.max_image_size = 100;
clusts_mouth_b = train_patch_classifier(conf2_t,clusters,nonPersonIds,'toSave',true,'suffix',...
    'clusts_mouth_b','overRideSave',true);

[qq_mouthb,q_mouthb,aps_mouthb] = applyToSet(conf2,clusts_mouth_b,aa1,[],'clusts_mouth_b100','disp_model',true,...
    'add_border',false,'override',false,'useLocation',Zsb,'dets',q_mouthb);
Zsb = createConsistencyMaps(qq_mouthb,[64 64],1:20);
figure,imshow(multiImage(jettify(Zsb)))

ttt =  top_labels(qq_mouthb(1).cluster_locs(:,11));
%t1 = visualizeLocs2(conf2,aa1,qq_mouthb(1).cluster_locs(ttt(1:100),:));
t1 = visualizeLocs2(conf2,aa1,qq_mouthb(1).cluster_locs(1:100,:));
figure,imshow(multiImage(t1,false,false))
figure,imshow(multiImage(t1,false,find(ttt)));
[A,AA] = visualizeClusters(conf,aa1,qq_mouthb(3),'add_border',...
      false,'nDetsPerCluster',100,'disp_model',true);  
figure,imshow(multiImage(AA))

%%
[prec,rec,aps_,T,M_mouth] = calc_aps(qq_mouth,top_labels,sum(test_labels));
[prec,rec,aps_,T,M_eyes] = calc_aps(qq_eye,top_labels,sum(test_labels));
[prec,rec,aps_,T,M_mouthb] = calc_aps(qq_mouthb,top_labels,sum(test_labels));
[prec_orig,rec_orig,aps_orig,T_orig,M_orig] = calc_aps(dets_top_train(1),test_labels);
M_orig = M_orig(dets_top_train(1).cluster_locs(:,11),:);
M = [M_mouth,M_mouthb,M_eyes];

%%
%%
% select a feature in each image.
[w b ap] = checkSVM(M,top_labels);
r = M*w-b;
plot(r);
[r,ir] = sort(r,'descend');
figure,imshow(multiImage(aa1(ir(1:50))))
[p_,r_,a_,T_] = calc_aps2(M*w,top_labels);
a_
plot(w)

%%
[prec,rec,aps_,T,M_f] = calc_aps(qq_frect2,top_labels,sum(test_labels));
[prec_orig,rec_orig,aps_orig,T_orig,M_orig] = calc_aps(dets_top_train(1),test_labels);
M_orig = M_orig(dets_top_train(1).cluster_locs(:,11),:);
%M = [M_mouth,M_mouthb,M_eyes,M_orig];
M = [M_f,M_orig];

%%
close all
% choice_ = 1:1000;

ff = find(top_labels);
ff = ff(1:20);
ff_ = find(~top_labels);
tt = [top_labels(ff);top_labels(ff_)];
MM = [M(ff,:);M(ff_,:)];

[w b ap] = checkSVM(MM,tt);
r = M*w-b;
plot(r);
[r,ir] = sort(r,'descend');
figure,imshow(multiImage(aa1(ir(1:50))))
[p_,r_,a_,T_] = calc_aps2(M*w,top_labels);
a_
plot(w)

%% combine orig - mouth
thetas = [1 0];
combined_mouth = combine_grades(qq_mouth,dets_top_train(1),thetas,[]);
[prec,rec,aps_,T,M_mouth] = calc_aps(combined_mouth,top_labels,sum(test_labels));
[r,ir] = sort(aps_,'descend');
disp(r);

%%
[prec_orig,rec_orig,aps_orig,T_orig,M_orig] = calc_aps(dets_top_train(1),test_labels);
M_orig = M_orig(dets_top_train(1).cluster_locs(:,11),:);
M = [M_orig,M_mouth];
%%
theta = [-1 1];
Z = bsxfun(@plus,theta(1)*M_mouth,theta(2)*M_orig);
% [z,iz] = sort(Z,'descend');
% figure,imagesc(iz);
[prec_z,rec_z,aps_z,T_z] = calc_aps2(Z,top_labels,sum(test_labels));
% figure,imagesc(T_z)
disp(sort(aps_z,'descend'));
[r,ir] = sort(aps_z,'descend');
%%
theta = [1 1];
Z_eye = bsxfun(@plus,theta(1)*Z(:,ir(1)),theta(2)*M_eyes);
[prec_z_eye,rec_z_eye,aps_z_eye,T_z_eye] = calc_aps2(Z_eye,top_labels,sum(test_labels));
disp(sort(aps_z_eye,'descend'))
[b,ib] = sort(aps_z_eye,'descend');
[q,iq] = sort(Z_eye(:,ib(1),:),'descend');

%%
figure,imshow(multiImage(aa1(iq(1:150))))
%%

plot(rec,prec)
[A1,AA1] = visualizeClusters(conf,aa1(1:1000),combined(ir(1)),'add_border',...
      false,'nDetsPerCluster',100,'disp_model',true);

  %%
  
Zs = createConsistencyMaps(qq_eye,[64 64],1:100);
figure,imagesc(Zs{1})
figure,imagesc(aa1{1})

is_true = (top_labels(qq_eye(1).cluster_locs(:,11)));

figure,imshow(multiImage(aa1(qq_eye(1).cluster_locs(is_true,11))));

find(is_true)
plot(qq_eye(1).cluster_locs(is_true,12))

% figure,imshow(multiImage(aa1(iq(1:150))))
%   aaa1 =visualizeLocs2(conf2,aa1,qq_eye(1).cluster_locs(top_labels,:),'inflateFactor',2,...
%       'height',64);
%   figure,imshow(multiImage(aaa1))

  %% 


