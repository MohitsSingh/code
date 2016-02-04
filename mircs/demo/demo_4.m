config;


load dets_top_train_fixed
load dets_top_test_fixed

[train_ids,train_labels] = getImageSet(conf,'train');

[test_ids,test_labels] = getImageSet(conf,'test');

train_inds_1= dets_top_train(1).cluster_locs(:,11);
true_dets = find(train_labels(train_inds_1));

p_train1 = visualizeLocs2(conf,train_ids,dets_top_train(1).cluster_locs(true_dets,:));

figure,imshow(multiImage(p_train1,true_dets));

% re-detect only on these images, and train, to find a better "head"
% detector....

conf.detection.params.detect_save_features = 1;
% 'conf,clusters,imgSet,labels,suffix,varargin)
matlabpool
redetect_1_train = applyToSet(conf, dets_top_train(1),train_ids(train_inds_1(true_dets)),...
    [],'redetect_1_train','override',false);
sel_ = 1:20;
redetect_new1 = makeCluster(cat(2,redetect_1_train.cluster_samples(:,sel_)),...
    cat(1,redetect_1_train.cluster_locs(sel_,:)));

nonPersonIds = getNonPersonIds(VOCopts);
redetect_new1_train = train_patch_classifier(conf,redetect_new1,nonPersonIds,'suffix','redetect_new1_train');

conf.detection.params.detect_save_features = 0;


redetect_new1_train_t = applyToSet(conf, redetect_new1_train,train_ids,...
   train_labels,'redetect_new1_train_t','override',false,'nDetsPerCluster',10);

redetect_new1_test_t = applyToSet(conf, redetect_new1_train,test_ids,...
   test_labels,'redetect_new1_train_test','override',false,'nDetsPerCluster',10);

t =find(test_labels(redetect_new1_test_t.cluster_locs(:,11)));
r = visualizeLocs2(conf,test_ids,...
    redetect_new1_test_t.cluster_locs(t(1:100),:));

t_neg =find(~test_labels(redetect_new1_test_t.cluster_locs(:,11)));
r_neg = visualizeLocs2(conf,test_ids,...
    redetect_new1_test_t.cluster_locs(t_neg(1:100),:));

figure,imshow(multiImage(r_neg))


r = visualizeLocs2(conf,train_ids,...
    redetect_new1_train_t.cluster_locs(1:100,:));

figure,imshow(multiImage(r))

plot(redetect_new1_train_t.cluster_locs(:,12))

imshow(showHOG(conf,redetect_new1_train_t))

% should make a very good head detector....
