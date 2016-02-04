% clear classes;
initpath;
config;
% load the clusters...
% conf_orig = conf;
% conf.suffix = 'train_mdf_tt';
% clusters = train_patch_classifier(conf,[],[],1);
% clusters = rmfield(clusters,'sv');
% load clusters_lite;

% precompute the cluster responses for the entire training set.
%
conf.suffix = 'train_dt_noperson';
conf.VOCopts = VOCopts;
% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);

% clusters = refineClusters(conf,clusters,discovery_sets,natural_set);


% for testing, start from second image and get every second image

% note - TODO maybe it is better at this stage to choose a random
% (balanced) subset of positive and negative examples, so each decision
% tree can be slightly different.

% since we're only testing now if the tree works, we'll work in fast 'n
% easy mode: small images, small dataset.

% conf.features.winsize = 8;
% conf.detection.params.detect_levels_per_octave = 3;
% conf.detection.params.detect_keep_threshold = -50; % keep also very low scoring results
% in order to make the info_gain computation "fair"
conf.detetion.params.detect_max_windows_per_exemplar = 1;

% % suffix = 'clusters_train_1';
% % matlabpool
% % [dets_train] = getDetections(conf,train_ids,clusters,1,suffix,1)
% % conf.clustering.top_k = length(train_ids);
% % %(conf,detections,clusters,uniqueImages)
% % top_dets_train = getTopDetections(conf,dets_train,clusters,1);
% % save('top_dets_train.mat','top_dets_train');
%
% x
% [prec_train,rec_train,aps_train,T_train,M_train] = calc_aps(top_dets_train,train_labels);
%

% save train_res prec_train rec_train aps_train T_train M_train;
%
load top_dets_train_lite
load train_res
[atrain,itrain] = sort(aps_train,'descend');
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);

% refine only the top detectors....
clusters_top =top_dets_train_lite;
conf.clustering.top_k = 5;

%%
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
ids = ids(t==-1); % persons non gratis
[discovery_sets,natural_set] = split_ids(conf,train_ids,train_labels);
natural_set = {ids(1:2:end),ids(2:2:end)};
%%
conf.suffix = 'train_dt_noperson'; 

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
    conf.suffix = ['train_dt_noperson_' num2str(ic)];
    refineClusters(conf,clusters_top(chunkStart:chunkEnd),...
        discovery_sets,natural_set,conf.suffix,'keepSV',false);
end

clusters = {};
for ic = 1:length(c)
    
    disp(100*ic/length(c))
    conf.suffix = ['train_dt_noperson_' num2str(ic)];
    clusters{ic} = train_patch_classifier(conf,[],[],conf.clustering.num_iter,'toSave',true);
end

clusters = cat(2,clusters{:});

matlabpool
suffix = 'clusters_all_test';
clusters = makeLight(clusters,'sv','vis','cluster_samples');
% save clusters_all_lite.mat cc
%(itrain(3));
% L = load('~/storage/data/cache/detectors_1train_dt_noperson_ftt');
[test_ids,test_labels] = getImageSet(conf,'test',1,0);
% test_ids = test_ids(1:100:end);
% test_ids = test_ids(test_labels);
% test_labels = test_labels(1:100:end);
% test_labels = test_labels(1:5);
[dets_test] = getDetections(conf,test_ids ,clusters,1,suffix,1);
conf.clustering.top_k = inf;
dets_test_top_ = getTopDetections(conf,dets_test,cc,1);
dets_test_top_ = makeLight(dets_test_top_,'sv','cluster_samples');

save clusters_all_1816_test.mat dets_test_top_

[prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(dets_test_top_,test_labels,[]);
[p,ip] = sort(aps_test,'descend');
figure,imagesc(T_test(:,ip(1:5)))
figure,plot(rec_test(:,ip(1)),prec_test(:,ip(1)))

% choose only a subset of the detectors.
M_test1 = M_test;
Tap =.1;
M_test1 = M_test1(:,ip(p>=Tap));
[M_test_,iM_test] = sort(M_test1);
iM_test =iM_test(1:50,:);
Q = zeros(size(M_test_));
for k = 1:size(Q,2)
    Q(iM_test(:,k),k) = 1;
end

chosenCols = 1;
curVec = Q(:,1);
Tc = 70; % iterate while adding new detectors diversifies images enough.

for c = 1:20
    curProd =curVec'*Q;    
    [minProd,imin] = min(curProd);
    disp(minProd)
    if (minProd > Tc)
        break; % nothing more to add
    end
    chosenCols = [chosenCols,imin];
    curVec = max(curVec,Q(:,imin));
end

% % % % ints_ = Q'*Q; % intersection matrix
% % % 
% % % % TODO - note that this is the intersection matrix for image indices,
% % % %. i.e, two detectors intersect at an image if they both fire on it (with a
% % % %rather high score).
% % % % a different (better?) way to do this would be to check if they actually
% % % % fire on the same image region (rectangle intersection).
% % % 
% % % ints_ = ints_(ip,:);
% % % ints_ = ints_(:,ip);
% % % q = ints_;
% % % 
% % % % threshold the allowed AP - don't allow it to be too small
% % % Tap =.1;
% % % q(:,p<=Tap) =inf;
% % % 
% % % % [q,iq] = sort(ints_,2,'ascend');
% % % Tc = 5; % iterate while adding new detectors diversifies images enough.
% % % 
% % % chosenSet = 1;
% % % % q(eye(length(p))>0)=inf; % make sure no one chooses itself
% % % % q(:,1) = inf; % already remove first element from optional choice
% % % c = 1;
% % % while c <= 50    
% % %     curChosen =sort(chosenSet,'ascend');    
% % %     % find for each element of the chosen set the element with the smallest
% % %     % intersection yet.
% % %     chosenRows = q(curChosen,:);
% % %     % choose best column for each row
% % %     [m,iCol] = min(chosenRows,[],2);
% % %     % find best row
% % %     [mm,iRow] = min(m);
% % %     minCol = iCol(iRow);
% % %     % add current row to chosen set?
% % %     if (mm > Tc) % did not find anyone do diversify enough, stop
% % %         break;
% % %     end
% % %     c = c+1;
% % %     chosenSet = [chosenSet,minCol]; 
% % %     % 
% % %     %chosenSet(minCol) = c;
% % %     q(:,minCol) = inf; % may not choose this again 
% % % end

[a,aa] = visualizeClusters(conf,test_ids,dets_test_top_(ip(chosenCols(1:20))),'height',64,...
    'disp_model',true,'add_border',true,'nDetsPerCluster',10);

imwrite(clusters2Images(a),'diversity3.jpg');

% choose some relativelty good scoring images.
% T = double(T_test(:,aps_test>=.10));
% [X,norms] = normalize_vec(T);
% inProd = (X'*X);
% [inProd,ii] = sort(inProd);
% pp = inProd(:,ip);
% pp = pp(ip,:);
% figure,imagesc(pp)
% % diversify the results greedily...
% 
% aps_ = aps_test(aps_test>=.10);
% [p_,ip_] = sort(aps_,'descend');



[a,aa] = visualizeClusters(conf,test_ids,dets_test_top_,'height',64,...
    'disp_model',true,'add_border',true,'nDetsPerCluster',50);
% ,64,1,1);

%%
suffix = 'clusters_all_train';
cc = clusters%(itrain(3));
% L = load('~/storage/data/cache/detectors_1train_dt_noperson_ftt');
[test_ids,test_labels] = getImageSet(conf,'test',1,0);
% test_ids = test_ids(1:100:end);
% test_ids = test_ids(test_labels);
% test_labels = test_labels(1:100:end);
% test_labels = test_labels(1:5);
[dets_test] = getDetections(conf,test_ids ,cc,1,suffix,1);
conf.clustering.top_k = inf;
dets_test_top_ = getTopDetections(conf,dets_test,cc,1);
dets_test_top_ = makeLight(dets_test_top_,'sv','cluster_samples');
%%

imwrite(multiImage(aa),'222_test.jpg');

m = clusters2Images(a);
imwrite(m,'111_test.jpg');
hist(dets_test_top_(3).cluster_locs(:,12))

[prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(dets_test_top_,test_labels);

plot(sort(aps_test))

[p,ip] = sort(aps_test,'descend');

plot(rec_test(:,ip(1)),prec_test(:,ip(1)))
[a,aa] = visualizeClusters(conf,test_ids,dets_test_top_(ip(1)),'height',64,...
    'disp_model',true,'add_border',true,'nDetsPerCluster',500);
f = find(test_labels(dets_test_top_(ip(1)).cluster_locs(:,11)));
figure,imshow(multiImage(aa(f(1:min(100,min(length(f),length(aa)))))))
figure,imshow(multiImage(aa))

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
dt.T_maxDepth =8;
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
dt.train_tree(train_ids,train_labels,M_train,dets_test_top_);

q = examineTree(dt)
q = cat(1,q{:});

%a = visualizeClusters(conf,train_ids,top_dets_train(itrain(1)),64,1,1);
a = visualizeClusters(conf,train_ids,dets_test_top_(q),64,1,1);
m = clusters2Images(a);
figure,imshow(m)
imwrite(m,'dt.jpg');
%%
% imshow(dt.rootNode.leftChild.debugInfo.goodPatch);
% % % imshow(dt.rootNode.debugInfo.goodPatch);
[test_ids,test_labels] = getImageSet(conf,'test',1,0);
[discovery_sets1,natural_set1] = split_ids(conf,test_ids,test_labels);
% %
dt.debugMode = false;
% for q = 1:10
%     res1(q) = dt.classify(discovery_sets1{1}(q));
% end
res1=(dt.classify(discovery_sets1{1}(1:50)));
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