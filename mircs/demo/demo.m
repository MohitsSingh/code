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
q_orig_t_d_test_g = det_union(q_orig_t_d_test([1 5]));
[prec_u,rec_u,aps_u,T_u,M_u]= calc_aps(q_orig_t_d_test_g,test_labels);
aps_u
%%
%% good - now we can try to do this for a couple of detectors (one for side-bottle,
% one for holding-cup

q_orig_t5= train_patch_classifier(conf,[],[],'toSave',true,'suffix',...
    'retrain_consistent1_d_5','overRideSave',false);

q_orig_t_d_test5 = applyToSet(conf,q_orig_t5,test_ids,test_labels,...
    'ids_reorder_test_d12_1_5','nDetsPerCluster',10,'override',true,'disp_model',true);
[prec,rec,aps,T,M]= calc_aps(q_orig_t_d_test5,test_labels);
plot(rec,prec)

[q_orig_t_d_test5_u] = det_union(q_orig_t_d_test5([1 2]));
[prec_,rec_,aps_,T_,M_]= calc_aps(q_orig_t_d_test5_u,test_labels);
% now unite with the first one...
u2 = cat(1,q_orig_t_d_test5_u,q_orig_t_d_test_g);
[u2] = det_union(u2);
[prec_,rec_,aps_,T_,M_]= calc_aps(u2,test_labels);
% excellent! 25%
plot(rec_,prec_)
[A,AA] = visualizeClusters(conf3,test_ids,u2,'add_border',...
    true,'nDetsPerCluster',50,'gt_labels',test_labels,...
    'disp_model',false,'interactive',false);

[A,AA] = visualizeClusters(conf3,test_ids,[q_orig_t_d_test5,q_orig_t_d_test_g],...
'add_border',true,'nDetsPerCluster',8,'gt_labels',test_labels,...
    'disp_model',false,'interactive',false,'height',64);

imwrite(clusters2Images(A),'mircs.jpg','quality',100);

figure,imshow(multiImage(AA))
imwrite(multiImage(AA),'mirc_union.jpg','quality',100);

% now try with the "regrade...."
% %%
% ii = q_orig_t_d_test(1).cluster_locs(:,11);
% u2_p = u2;
% [a,b,c] = intersect(u2.cluster_locs(:,11),ii);
% u2_p.cluster_locs = u2_p.cluster_locs(b,:);

%%
for theta = .5
    re_graded = combine_grades(clusts_smirc_test_check,u2_p,theta,[]);
%     % re_graded = combine_grades(clusts_smirc_test_check(3),q_orig_t_d_test(1),theta,the_labels);
    [prec,rec,aps_,T,M] = calc_aps(re_graded,the_labels,sum(test_labels));
    [r,ir] = sort(aps_,'descend');
    disp(r(1:5));
    %
end


