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

conf.detetion.params.detect_max_windows_per_exemplar = 1;

load top_dets_train_lite
load train_res
[atrain,itrain] = sort(aps_train,'descend');
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);

% refine only the top detectors....
clusters_top =top_dets_train_lite(itrain(1:20));
conf.clustering.top_k = 5;

%%
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
ids = ids(t==-1); % persons non gratis
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
natural_set = {ids(1:2:end),ids(2:2:end)};
%%
conf.suffix = 'train_dt_noperson_top'; 

for k = 1:length(clusters_top)
    clusters_top(k).w = clusters_top(k).cluster_samples(:,1);
    clusters_top(k).cluster_samples= clusters_top(k).cluster_samples(:,1);
    clusters_top(k).b = 0;
end
conf.clustering.num_iter = 5;

nClustersPerChunk = 50;
c = 1:nClustersPerChunk:length(clusters_top);

for ic = 1:length(c)
    chunkStart = c(ic);
    if (ic == length(c))
        chunkEnd = length(clusters_top);
    else
        chunkEnd = c(ic+1)-1;
    end
    conf.suffix = ['train_dt_noperson_top_' num2str(ic)];
    clusters_top = refineClusters(conf,clusters_top(chunkStart:chunkEnd),...
        discovery_sets,natural_set,conf.suffix,'keepSV',false);
end

% matlabpool

chooseSet = 'train';
suffix = ['clusters_' chooseSet '_1_np_pos1_top'];
cc = clusters_top%(itrain(3));
% L = load('~/storage/data/cache/detectors_1train_dt_noperson_ftt');
[test_ids,test_labels] = getImageSet(conf,chooseSet,1,0);
% test_ids = test_ids(1:100:end);
% test_ids = test_ids(test_labels);
% test_labels = test_labels(1:100:end);
% test_labels = test_labels(train_labels);
toSave = 1;
[dets_test] = getDetections(conf,test_ids ,cc,1,suffix,toSave);
conf.clustering.top_k = inf;
dets_test_top_ = getTopDetections(conf,dets_test,cc,1);
[prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(dets_test_top_,test_labels);
plot(sort(aps_test))
[a,aa] = visualizeClusters(conf,test_ids,dets_test_top_,'height',64,...
    'disp_model',true,'add_border',true,'nDetsPerCluster',20);

m = clusters2Images(a);
imwrite(m,'333_test1.jpg');

dets_top_test = dets_test_top_;
% now apply only this detector to the training images...
%%
% cur_ids = test_ids(test_labels);
cur_ids = test_ids;
cur_labels = test_labels;
cur_cluster = cc(1);
cur_top_dets = dets_test_top_(1);
% [cur_det] = getDetections(conf,test_ids ,cur_cluster,1,suffix,0);
%conf.clustering.top_k = inf;
%cur_top_dets = getTopDetections(conf,cur_det,cur_cluster,1);
% [a,aa] = visualizeClusters(conf,cur_ids,cur_top_dets,'height',64,...
%     'disp_model',true,'add_border',true,'nDetsPerCluster',inf);

aa1 = visualizeLocs2(conf,test_ids,cur_top_dets.cluster_locs(:,:),'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false);

save top_20_1_patches aa1

conf_orig = conf;
conf2 = conf_orig;
conf2.features.winsize = 4;
conf2.suffix = strrep(conf2.suffix,'_1','_2');
conf2.clustering.top_k = 5;
cur_labels= test_labels(cur_top_dets.cluster_locs(:,11));
true_set = aa1(cur_labels);
false_set =aa1(~cur_labels);
discovery_sets = {true_set(1:2:end),true_set(2:2:end)};
natural_sets = {false_set(1:2:end),false_set(2:2:end)};
conf2.detection.params.detect_min_scale = .8;
clustering3(conf2,discovery_sets,natural_sets,'ovp',.2,'clusteringConf',conf2);

p = '/home/amirro/storage/data/cache/detectors_5train_dt_noperson_top_2';
clusters = rmfield(L.clusters,'sv');
clusters = rmfield(clusters,'cluster_samples');

save('clusters_2_lite.mat','clusters');

%% get corresponding test set detector...

dets_test_top_ = getTopDetections(conf,dets_test,cc,1);

subdet = dets_test_top_(3);
y = test_labels(subdet.cluster_locs(:,11));
% subdet.cluster_locs = subdet.cluster_locs(1:300,:);
aa2 = visualizeLocs2(conf,test_ids,subdet.cluster_locs,'add_border',true,...
    'inflateFactor',1.5,'height',96,'draw_rect',false);


conf_new = conf;
conf_new.features.winsize = 4;
% apply all detectors to this...
new_det = getDetections(conf_new,aa2,clusters,[],[],0);
cur_top_dets1 = getTopDetections(conf_new,new_det,clusters,1);

[prec_test2,rec_test2,aps_test2,T_test2,M_test2] = calc_aps(cur_top_dets1,y);

figure,plot(sort(aps_test2))
[pp,ipp] = sort(aps_test2,'descend');

figure,plot(rec_test2(:,ipp(1:10)),prec_test2(:,ipp(1:10)))

aa3 = visualizeLocs2(conf_new,aa2,cur_top_dets1(ipp(10)).cluster_locs(1:50,:),'add_border',false,...
    'inflateFactor',1,'height',32,'draw_rect',false);

figure,imshow(multiImage(aa3))

cur_top_dets1


cur_cluster = cc(3);
[cur_det] = getDetections(conf,test_ids,cur_cluster,1,suffix,0);

[test_ids,test_labels] = getImageSet(conf,'test',1,0);
cur_ids = test_ids;
cur_labels = test_labels;
cur_cluster = cc(3);
[cur_det] = getDetections(conf,test_ids,cur_cluster,1,suffix,0);
conf.clustering.top_k = inf;
cur_top_dets = getTopDetections(conf,cur_det,cur_cluster,1);
% [a,aa] = visualizeClusters(conf,cur_ids,cur_top_dets,'height',64,...
%     'disp_model',true,'add_border',true,'nDetsPerCluster',inf);

aa1 = visualizeLocs2(conf,test_ids,cur_top_dets.cluster_locs(:,:),'add_border',false,...
    'inflateFactor',1.5,'height',96,'draw_rect',false);


% chooseSet ='test';
% [test_ids,test_labels] = getImageSet(conf,chooseSet,1,0);
% toSave = 1;
% matlabpool
% test_ids = test_ids(1:100:end);
% % test_ids = test_ids(test_labels);
% test_labels = test_labels(1:100:end);
% % test_labels = test_labels(train_labels);
% suffix = 'test_2';
% conf_new = conf;
% conf_new.features.winsize = 4;
% [dets_test2] = getDetections(conf_new,test_ids ,clusters,1,suffix,toSave);
% conf_new.clustering.top_k = inf;
% dets_test_top_2 = getTopDetections(conf_new,dets_test2,clusters,1);
% 
% aa1 = visualizeLocs2(conf,test_ids,dets_test_top_2(500).cluster_locs(1:50,:),'add_border',false,...
%     'inflateFactor',1,'height',32,'draw_rect',false);

figure,imshow(multiImage(aa1))


% allF = zeros(1984,length(aa));
% for k = 1:length(aa)
%     k
%     x = allFeatures(conf,aa{k});
%     allF(:,k) = x;
% end
% ttt = 50;
% y = test_labels();
% 
% % figure,imshow(multiImage(aa(y(1:10))))
% 
% checkSVM(allF(:,1:ttt)',y);
%%

tt = dets_test_top_(3);

tt.cluster_locs = tt.cluster_locs(1:10:end,:);
[a,aa] = visualizeClusters(conf,test_ids,tt,'height',64,...
    'disp_model',true,'add_border',true,'nDetsPerCluster',inf);

% ,64,1,1);
% dd = dets_test_top_(1);
% dd_c = dd.cluster_locs(:,11);
% [z,iz] = sort(dd_c);
% dd.cluster_locs = dd.cluster_locs(iz,:);
% [a,aa] = visualizeClusters(conf,test_ids,dd,'height',64,...
%     'disp_model',true,'add_border',true,'nDetsPerCluster',50);
imwrite(multiImage(aa),'many_test.jpg');

m = clusters2Images(a);
imwrite(m,'222_test.jpg');
hist(dets_test_top_(3).cluster_locs(:,12))

[prec_,rec_test,aps_test,T_test,M_test] = calc_aps(dets_test_top_,test_labels);

plot(rec_test,prec_test)
% figure,imagesc(T_test)
% sum(T_test)

[dets_test_top] = getDetections(conf,train_ids,clusters_top,[],[],1);
conf.clustering.top_k = inf;
dets_test_top_ = getTopDetections(conf,dets_test_top,clusters_top,1);
[prec_train,rec_train,aps_train,T_train,M_train] = calc_aps(dets_test_top_,train_labels);

% figure,imshow(m)

%top_dets_train_lite = top_dets_trainrmfield(top_dets_train,'cluster_locs');
% save top_dets_train_lite top_dets_train_lite
% load top_dets_train_lite
% load train_res
%  [atrain,itrain] = sort(aps_train,'descend');

%%

dt = decision_tree(conf);
dt.T_maxDepth =2;
% dbstop in tree_node at 293;
dbstop in tree_node at 40;
dbstop in decision_tree at 28;
% dbstop in result_set at 21;
% % % dbstop in decision_tree at 105;
% dbstop in allFeatures at 80;
dbstop if error
% a = visualizeClusters(conf,test_ids,top_dets_test(iaa(1:20)),64,1,1);

% top_choice = 20;
% tt = itrain(1:top_choice);
% top_dets_train = top_dets_train(tt);
% M_train =M_train(:,tt);
dt.train_tree(train_ids,train_labels,M_train(:,1:end),dets_test_top_);

q = examineTree(dt)
q = cat(1,q{:});

%a = visualizeClusters(conf,train_ids,top_dets_train(itrain(1)),64,1,1);
a = visualizeClusters(conf,train_ids,dets_test_top_(q));
m = clusters2Images(a);
figure,imshow(m)
imwrite(m,'dt.jpg');
%%
% imshow(dt.rootNode.leftChild.debugInfo.goodPatch);
% % % imshow(dt.rootNode.debugInfo.goodPatch);
[test_ids,test_labels] = getImageSet(conf,'test',1,0);
[discovery_sets1,natural_set1] = split_ids(conf,test_ids,test_labels);
% %
dt.debugMode = true;
% for q = 1:10
%     res1(q) = dt.classify(discovery_sets1{1}(q));
% end
res1=(dt.classify(discovery_sets1{1}(60:63)));
% res2=(dt.classify(natural_set1{1}(1:50)));

hist(res2)

getImagePath(conf,discovery_sets1{1}{1})

% %%
% addpath(genpath('/home/amirro/code/3rdparty/object_bank/MATLAB_release/code/partless/'));
% tt =find(~test_labels);
% for q =1:length(tt)
%     curImagePath = getImagePath(conf,test_ids{tt(q)});
%     getfeat_single_image(curImagePath);
%     pause;
% end
%%

res1 = dt.classify(test_ids);

%
% [prec,rec,aps,T] = calc_aps2(res1(:),test_labels));

res2 = dt.classify(test_labels);
% end
% %
% % for q = 1:10
% %     res2 = dt.classify(natural_set1{1}(q));
% % end
% %
% % gt_labels = zeros(length(res1)+length(res2),1);
% % gt_labels(1:length(res1)) = 1;
% %
% % rr = [res1(:);res2(:)];
% % p = randperm(length(rr));
% % rr = rr(p);
% % gt_labels = gt_labels(p);
% %
% % [prec,rec,aps,T] = calc_aps2(rr,gt_labels');
% %
% % for q = 1002:100:2000
% %     res2 = dt.classify(natural_set1{2}(q))
% % end
% %
% %
% % % % %
% % % recursively show the tree...
% % toExplore = dt.rootNode;
% % str= {'middle'};
% % while (~isempty(toExplore))
% %     curNode = toExplore(1);
% %     toExplore = toExplore(2:end);
% %     curTitle = str{1};
% %     str =str(2:end);
% %     if (curNode.isLeaf)
% %         continue;
% %     end
% %     %     if (curNode.leftChild.isValid)
% %     str{end+1} ='left';
% %     toExplore = [toExplore,curNode.leftChild];
% %     %     end
% %     %     if (curNode.rightChild.isValid)
% %     str{end+1} = 'right';
% %     toExplore = [toExplore,curNode.rightChild];
% %     %     end
% %     imshow(curNode.debugInfo.goodPatch);
% %     title(curTitle);
% %     pause;
% % end
% % % % % %
% % % % % % conf.max_image_size = 200;
% % % % % % conf.features.winsize = 4;
% % % % % % conf.detection.params.detect_min_scale = .25;
% % % % % % I1 = getImage(conf,natural_set1{2}{1});
% % % % % % I2 = getImage(conf,natural_set1{2}{2});
% % % % % %
% % % % % % X1 =allFeatures(conf,I1);
% % % % % % X2 =allFeatures(conf,I2);
% % % % % % size(X1)
% % % % % % addpath /home/amirro/code/3rdparty/lshcode
% % % % % % T = lsh('lsh',5,24,size(X1,1),X1);
% % % % % %
% % % % % % lshstats(T,'test',X1,X2,2)
% % % % % % tic
% % % % % % for q = 1:size(X2,2)
% % % % % % [iNN,cand] = lshlookup(X2(:,q),X1,T);
% % % % % % end
% % % % % % toc
% % % % % %
% % % % % % a = l2(X1',X2');
% % % % % % figure,imagesc(a)
% % % % % % lshstats(T)
% % % % %
% % % % %
% % %% new scratch
% %
% % % after loading this, we have cluster with the detection scores for
% % % (almost) each image in the training set.
% %
% % conf.clustering.top_k = length(test_ids);
% % %(conf,detections,clusters,uniqueImages)
% % top_dets_test = getTopDetections(conf,dets_test,clusters,1);
% % save('top_dets_test.mat','top_dets_test');
% %
% % [prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(top_dets_test,test_labels);
% % save data_test prec_test rec_test aps_test T_test M_test
% % [atest,iatest] = sort(aps_test,'descend');
% % a = visualizeClusters(conf,test_ids,top_dets_test(iaa(1:20)),64,1,1);
% % % a = visualizeClusters(conf,train_ids,top_dets_train(1:20:end),64,1,1);
% % m = clusters2Images(a);
% % imwrite(m,'demo2_test_ap_bytrain.jpg');
% %
% %
% % load top_dets_train.mat;


%%

% % % attempt to cluster according to SVM weights!!
% % ww = cat(2,top_dets_train.w);
% % b = cat(2,top_dets_train.b);
% % ww = [ww;b];
% % figure,imagesc(ww);
% % nClusters = size(ww,2)/5;
% % [C,A] = vl_kmeans(ww,nClusters,'Algorithm','lloyd');
% %
% % figure,imagesc(C)
% %
% % % now get the top 1 det from each cluster....
% % clusters_united = initClusters;
% % for k = 1:size(C,2)
% %     clusters_united(k).w = C(1:end-1,k);
% %     t = find(A==k);
% %     cluster_locs = zeros(length(t),12);
% %     for q = 1:length(t)
% %         cluster_locs(q,:) = top_dets_train(t(q)).cluster_locs(1,:);
% %     end
% %     clusters_united(k).cluster_locs = cluster_locs;
% %     clusters_united(k).isvalid = true;
% % end
% %
% % a = visualizeClusters(conf,train_ids,clusters_united(1:end),64,1,1);
% % m = clusters2Images(a);
% % imwrite(m,'united_b.jpg');