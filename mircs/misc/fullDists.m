% get a set of training images...
initpath; 
config2;
conf.suffix = 'train_s_4';
conf.class_subset = DRINKING;
% dataset of images with ground-truth annotations
% start with a one-class case.
[train_ids,train_labels] = getImageSet(conf,'train',2,0);
[val_ids,val_labels] = getImageSet(conf,'train',2,1);
% conf.detection.params.init_params.sbin = conf.features.sbin;
conf.features.winsize = 8;
% % limit image size for slightly faster calculations (at the
% % cost of less accurate results? I need to check)
conf.max_image_size = 256;
conf.detection.params.detect_min_scale = .5;
conf.detection.params.detect_levels_per_octave = 10;
%  
conf.clustering.min_cluster_size = 1;
conf.clustering.split_discovery = false;

%%
conf.clustering.max_sample_ovp = .5;
% conf.clustering.windows_per_image = inf;
ids_true_train = col(train_ids(train_labels))';
% ids_true_train = ids_true_train(1:20);
ids_false_train = vl_colsubset(col(train_ids(~train_labels))',length(ids_true_train),'Uniform');
% ids_false_train = ids_false_train(1:20);
% 
% dists12 = imageSetDistances(conf,ids_true_train,ids_false_train);
load dists12;
%%
[samples,locs,dists] = findDiscriminativePatches(conf,ids_true_train,ids_false_train,dists12,0);
% [nn_samples,nn_locs,v] = findNearestPatches(conf,samples,ids_true_train,1,dists);

%visualizeLocs2(conf,ids,locs,height,inflateFactor,add_border,draw_rect)
locs_ = cat(1,locs{:});


[p] = visualizeLocs2(conf,ids_true_train,locs_(:,:),64,1,0,0);
% close all;
% figure,imshow(multiImage(p(1:end)));
suffix = 'mdf_1_8_1';
conf_n = conf;
% locs = locs(1:10);
% samples = samples(1:10);
conf.detection.params.detect_levels_per_octave = 10;
conf.detection.params.detect_add_flip = 0;

samples_cat = cat(2,samples{:});

[neighbors,allLocs,allFeats] = findNeighbors2(conf,samples_cat,ids_true_train,suffix,0,1);
conf.clustering.top_k = 1;
clusters = neighbors2clusters(conf,neighbors,allLocs,allFeats,true);

[clusters,allP] = visualizeClusters(conf,ids_true_train,clusters,64);
m = clusters2Images(clusters,[0 0 0]);
% imshow(m)
imwrite(m,'mdf_clusters_8_1_flip_resort.jpg','Quality',100);

% % consistency check...
% samples_ = clusters(1).cluster_samples;
% locs_ = clusters(1).cluster_locs;
% conf_t = conf;
% conf_t.suffix = 'train_s';
% dict = learnDictionary(conf_t,train_ids);
% conf.dict = dict;
% 
% 
% [neighbors1,allLocs1,allFeats1] = findNeighbors2(conf,samples_,ids_true_train,suffix,0,1);
% conf.clustering.top_k = 15;
% clusters1 = neighbors2clusters(conf,neighbors1,allLocs1,allFeats1,false);
% [clusters1,allP1] = visualizeClusters(conf,ids_true_train,clusters1,128);
% 
% figure,imshow(clusters2Images(clusters1))
% figure,imshow(clusters(1).vis)
% % sort clusters by mean distance to center...
% 
% d = zeros(size(clusters1));
% for k = 1:length(clusters1)
%     d(k) = mean(l2(clusters1(k).cluster_center',clusters1(k).cluster_samples(:,1:5)').^.5);
% end
% [dd,idd] = sort(d,'ascend');
% m = clusters2Images(clusters1(idd),[0 0 0]);
% imwrite(m,'mdf_clusters_8_1_s.jpg','Quality',100);
% 
% % 
conf.clustering.num_hard_mining_iters = 5;
clusters1 = train_patch_classifier(conf,clusters,train_ids(~train_labels),1,'_disc_8_resorted_2');
 
% % % res_suffix = 'mdf_8_res_val2';
% % % iter_num = [];
% % % 
% % % conf.detection.params.detect_add_flip = 1;
% % % conf.detection.params.detect_max_windows_per_exemplar = 10;
% % % conf.detection.params.detect_min_scale = .5;
% % % t = find(val_labels);
% % % t_not = find(~val_labels);
% % % tt = [t;t_not(1:end)];
% % % val_ids_ = val_ids(tt);
% % % val_labels_ = val_labels(tt);
% % % [top_dets,aps] = test_clusters(conf,clusters1,iter_num,suffix,val_ids_,val_labels_,res_suffix);
 %% sanity...
 res_suffix = 'mdf_8_sanity1'
iter_num = [];

conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_max_windows_per_exemplar = 1;
conf.detection.params.detect_min_scale = .5;
% t = find(val_labels);
% t_not = find(~val_labels);
% tt = [t;t_not(1:end)];
val_ids_ = train_ids(train_labels);
val_labels_ = train_labels(train_labels);
[top_dets,aps] = test_clusters(conf,clusters1,iter_num,suffix,val_ids_,val_labels_,res_suffix)
 

%  imwrite([m,addborder(imread('r.jpg'),3,[255 255 0])],'rr.jpg');
