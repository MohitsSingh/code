%     'applauding'
%     'blowing_bubbles'
%     'brushing_teeth'
%     'cleaning_the_floor'
%     'climbing'
%     'cooking'
%     'cutting_trees'
%     'cutting_vegetables'
%     'drinking'
%     'feeding_a_horse'
%     'fishing'
%     'fixing_a_bike'
%     'fixing_a_car'
%     'gardening'
%     'holding_an_umbrella'
%     'jumping'
%     'looking_through_a_microscope'
%     'looking_through_a_telescope'
%     'playing_guitar'
%     'playing_violin'
%     'pouring_liquid'
%     'pushing_a_cart'
%     'reading'
%     'phoning'
%     'riding_a_bike'
%     'riding_a_horse'
%     'rowing_a_boat'
%     'running'
%     'shooting_an_arrow'
%     'smoking'
%     'taking_photos'
%     'texting_message'
%     'throwing_frisby'
%     'using_a_computer'
%     'walking_the_dog'
%     'washing_dishes'
%     'watching_TV'
%     'waving_hands'
%     'writing_on_a_board'
%     'writing_on_a_book'
% get a set of training images...
initpath;
config2;
conf.suffix = 'train_s_4';
conf.class_subset = 36;
% dataset of images with ground-truth annotations
% start with a one-class case.
[train_ids,train_labels] = getImageSet(conf,'train',2,0);
[val_ids,val_labels] = getImageSet(conf,'train',2,1);
% conf.detection.params.init_params.sbin = conf.features.sbin;
conf.features.winsize = 4;
conf.detection.params.init_params.sbin =4;
% % limit image size for slightly faster calculations (at the
% % cost of less accurate results? I need to check)
conf.max_image_size = 128;
conf.detection.params.detect_min_scale = .5;
conf.detection.params.detect_levels_per_octave = 4;
%  
conf.clustering.min_cluster_size = 1;
conf.clustering.split_discovery = false;

%%
conf.clustering.max_sample_ovp = .5;
% conf.clustering.windows_per_image = inf;
ids_true_train = col(train_ids(train_labels))';
ids_true_train = ids_true_train(1:20);
ids_false_train = vl_colsubset(col(train_ids(~train_labels))',length(ids_true_train),'Uniform');
ids_false_train = ids_false_train(1:20);
% 
dists12 = imageSetDistances(conf,ids_true_train,ids_false_train);
%dists11 = imageSetDistances(conf,ids_true_train,ids_true_train);

%save dists_all dists12 dists11
%%
[samples,locs,dists] = findDiscriminativePatches(conf,ids_true_train,ids_false_train,dists12,0);
% [nn_samples,nn_locs,v] = findNearestPatches(conf,samples,ids_true_train,1,dists);

%visualizeLocs2(conf,ids,locs,height,inflateFactor,add_border,draw_rect)
[p] = visualizeLocs2(conf,ids_true_train,cat(1,locs{:}),64,1,0,0);
close all;
figure,imshow(multiImage(p(1:end)));
suffix = 'mdf_1_8_4';
conf_n = conf;
[clusters] = findNeighbours(conf,samples,locs,ids_true_train,suffix)

[clusters,allP] = visualizeClusters(conf,ids_true_train,clusters,64);

m = clusters2Images(clusters,[0 0 0]);
imwrite(m,'mdf_clusters_8_4.jpg');

% 
ids_= {};
for k = 1:length(ids_true_train)
    ids_{k} = toImage(conf,getImagePath(conf,ids_true_train{k}));
end
[p,p_mask] = visualizeLocs(conf,ids_,locs,64,1,false);
mmm = multiImage(p);
figure,imshow(mmm);
% figure,imshow(multiImage(v))
m = clusters2Images(clusters);

conf.clustering.num_hard_mining_iters = 5;
clusters = train_patch_classifier(conf,clusters,ids_false_train,1,'_disc');
 [clusters,allImgs] = visualizeClusters(conf,ids_true_train,clusters,64);
 
%  dets = 
 
res_suffix = 'mdf_1_res';
iter_num = [];
 [top_dets,aps] = test_clusters(conf,clusters,iter_num,suffix,val_ids,val_labels,res_suffix)


