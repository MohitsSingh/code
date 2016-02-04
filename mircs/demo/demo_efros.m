initpath;
config;
% conf.max_image_size = 200;
conf.suffix = 'train_efros';
% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);
% [val_ids,val_labels] = getImageSet(conf,'train',2,1);
% for testing, start from second image and get every second image
[discovery_sets,natural_sets] = split_ids(conf,train_ids,train_labels);

% create a k-means dictionary for the sampling process
dict = learnDictionary(conf,train_ids);
conf.dict = dict;

clustering(conf,discovery_sets,natural_sets,dict);

% checkClusterConsistency(conf);

% find "good" clusters on training set

[test_ids,test_labels] = getImageSet(conf,'test',1,0);

iter_num = 5;
clusters = [];
[top_dets,aps] = test_clusters(conf,clusters,iter_num,conf.suffix,test_ids,test_labels,'efros_test',inf);

[prec,rec,aps] = calc_aps(top_dets,val_labels,[],inf);
  [~,ibest] = sort(aps,'descend');
    aa = ibest(1:min(1,length(ibest)));
%     aa =1:20;
    dddd = 64;
    add_border = 1;
    conf.clustering.top_k = inf;
    [cc]= visualizeClusters(conf,test_ids,top_dets(aa),dddd,add_border);        
    ff = clusters2Images(cc);
[~,ibest] =sort(aps,'descend');


top_dets_trainval = top_dets(ibest(1:20));

k_choice = 1:10
clusters_ = {};
for k =k_choice%max_k%(top_dets)    
% k = 2
    suffix = [num2str(k) '_efros']
%     k = 7;    
    clusters{k} = secondaryClustering_mdf(conf,top_dets_trainval(k),test_ids,test_labels,suffix);
end

k = 1;
suffix = ['_discxy_128_1' num2str(k) '_efros'];
conf_new = get_secondary_clustering_conf(conf);
conf_new.detection.params.detect_add_flip = 1;
clusters = train_patch_classifier(conf_new,[],[],1,suffix);
[val_ids,val_labels] = getImageSet(conf,'train',1,0);
[new_ids,new_labels] = diluteSet(val_ids,val_labels,1,10);
conf_new.detection.params.detect_max_scale = 1;
% conf_new.detection.params.detect_min_scale = .8;
[top_dets,aps] = test_clusters(conf_new,clusters,[],[],new_ids,new_labels,'smirc_test_2',inf);

%% 

conf_new = get_secondary_clustering_conf(conf);

% 
% for k =k_choice
%     suffix = ['_discxy_128_1' num2str(k)];
%     c_ = train_patch_classifier(conf_new,clusters_,[],1,suffix);
%     clusters_{k} = c_([c_.isvalid]);
% end
% 
% ims = {};
% for k = k_choice
%     suffix = ['d4_im_full_train' num2str(k)];
%     [re_sorted_,prec_,rec_,aps_,dets_,im] = sort_secondary(conf_new,top_dets_trainval(k),clusters_{k},train_ids,train_labels,toshow,suffix);
%     ims{k} = im;
% end

%% test all
%% and test on another part!
[test_ids,test_labels] = getImageSet(conf,'test',1,0);
% make it a small test...
f_true = find(test_labels);
f_false = find(~test_labels);
f_false = f_false(1:end);
test_ids = [test_ids(f_true(:));test_ids(f_false)];
test_labels = [test_labels(f_true(:));test_labels(f_false)];

second_suffix = 'test_secondary_full_4';
%conf,clusters,iter_num,conf.suffix,val_ids,val_labels,'train_val');
[top_dets_test,aps_test] = test_clusters(conf,top_dets,iter_num,conf.suffix,test_ids,test_labels,second_suffix);
[bb,ibb] = sort(aps_test,'descend');
% %% choose the top 5 from each detection...
% new_dets = top_dets_test(1);
% d = 5000;
% new_locs = [];
% for k = 1:7
%     new_locs = [new_locs; top_dets_test(ibb(k)).cluster_locs(:,:)];
% end
%
% new_dets.cluster_locs = new_locs;
% [prec5,rec5,aps5] = calc_aps(new_dets,test_labels,[]);
% figure,plot(rec5,prec5)


%%
re_sorted = struct;

for k =k_choice
    close all;
    toshow = 0;
    suffix = ['d4_im_full' num2str(k)];
    [re_sorted_,prec_,rec_,aps_,dets_] = sort_secondary(conf_new,top_dets_test(k),clusters_{k},test_ids,test_labels,toshow,suffix,ims{k});
    % and check the results
    [prec2,rec2,aps2] = calc_aps(dets_,test_labels);
    [prec1,rec1,aps1] = calc_aps(top_dets_test(k),test_labels);
    figure,plot(rec2,prec2); hold on;plot(rec1,prec1,'r');
    legend({['phase 2; ap=' num2str(aps2,'%1.3f')],['phase 1; ap=' num2str(aps1,'%1.3f')]});
    re_sorted(k).dets_ = dets_;
    %     pause;
    
end

%% choose the top from each classifier first....
new_dets = dets_;
d = 5000;
new_locs = [];

tt = 1;

for q = 1:tt
    for k = k_choice
        p = re_sorted(k).dets_.cluster_locs(q,:);
        p(:,12) = 2*tt-q;
        new_locs = [new_locs; p];
    end
end

for k = k_choice
    p = re_sorted(k).dets_.cluster_locs(tt+1:end,:);
    %             p(:,12) = 2*tt-q;
    new_locs = [new_locs; p];
end

new_dets.cluster_locs = new_locs;
[prec5,rec5,aps5] = calc_aps(new_dets,test_labels,[]);
figure,plot(rec5,prec5); title(num2str(aps5));
