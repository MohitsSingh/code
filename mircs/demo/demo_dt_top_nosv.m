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
baseSuffix = 'train_noperson_top_nosv';
conf.suffix = baseSuffix;

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
    conf.suffix = [baseSuffix '_' num2str(ic)];
    
    clusters_top = refineClusters(conf,clusters_top(chunkStart:chunkEnd),...
        discovery_sets,natural_set,conf.suffix,'keepSV',false);
end

% matlabpool

chooseSet = 'test';
suffix = ['clusters_new' chooseSet '_nosv'];
cc = makeLight(clusters_top,'sv','vis','cluster_samples');
% L = load('~/storage/data/cache/detectors_1train_dt_noperson_ftt');
[test_ids,test_labels] = getImageSet(conf,chooseSet,1,0);
% test_ids = test_ids(1:100:end);
% test_ids = test_ids(test_labels);
% test_labels = test_labels(1:100:end);
% test_labels = test_labels(train_labels);
toSave = 1;
[dets_test] = getDetections(conf,test_ids ,cc,1,suffix,toSave);
conf.clustering.top_k = inf;
dets_test_top_ = getTopDetections(conf,dets_test,cc);
[prec_test,rec_test,aps_test,T_test,M_test] = calc_aps(dets_test_top_,test_labels);
plot(sort(aps_test))
% [a,aa] = visualizeClusters(conf,test_ids,dets_test_top_,'height',64,...
%     'disp_model',true,'add_border',true,'nDetsPerCluster',20);
% [r,ir] = sort(aps_test,'descend');
% m = clusters2Images(a);
% imwrite(m,['333_' chooseSet '.jpg']);

dets_top_test = dets_test_top_;
dets_top_train = dets_top_test;
% save dets_top_test dets_top_test
% now apply only this detector to the training images...
%%
% cur_ids = test_ids(test_labels);
cur_ids = test_ids;
cur_labels = test_labels;
cur_cluster = cc(1);
cur_top_dets = dets_test_top_(1);

% aa1 = visualizeLocs2(conf,test_ids,cur_top_dets.cluster_locs(:,:),'add_border',false,...
%     'inflateFactor',1,'height',64,'draw_rect',false,'saveMemory',true);
% save top_20_1_train_patches aa1
load top_20_1_train_patches
% do a secondary clustering on these patches. Start from a small number
% of ground truth faces.
gt_labels = cur_labels(cur_top_dets.cluster_locs(:,11));

aa_true = aa1(gt_labels);
aa_false = aa1(~gt_labels);

conf_new = conf;
conf_new.detection.params.init_params.sbin =2;
conf_sample = conf_new;
conf_sample.detection.params.detect_min_scale = 1;
% start from only a single image!
[samples_true,locs] = samplePatches(conf_new,aa_true(1),0);
q = visualizeLocs2(conf_new,aa_true,locs);
figure,imshow(multiImage(q));
% train this patch...
nonPersonIds = getNonPersonIds(VOCopts);
clusts = makeClusters(samples_true,locs);

% resize the pascal non-person images to something smaller to make
% this more feasible...
conf_new.max_image_size = 100;
clusts_trained = train_patch_classifier(conf_new,clusts,nonPersonIds,'toSave',true,'suffix',...
    'clusters_drinking_const1','overRideSave',false);
conf_new.max_image_size = conf.max_image_size;
suffix = 'clusts_apply_train_1';

matlabpool
qq_1 = applyToSet(conf_new,clusts_trained,aa1,col(true(size(aa1))),suffix,'disp_model',true,...
    'add_border',true,'override',false);

% 1. we can improve the classifiers with high consistency.
% 2. we can prune out the detections which are not consistent!

% sel_train = vl_colsubset(1:length(aa1),100,'Random');
sel_train = 1:200;

Zs = createConsistencyMaps(qq_1,size(aa1{1}),sel_train);

% sort by entropy...
ents =zeros(size(Zs));
for  k = 1:length(Zs)
    k
    curZ = Zs{k};
    ents(k) = entropy(curZ);
end
[e,ie] = sort(ents,'ascend');

for k = 1:length(qq_1)
    figure(1);imagesc(Zs{ie(k)});pause;
end


ZZ = cat(3,Zs{:});
ZZ = shiftdim(ZZ,2);
ZZ = reshape(ZZ,size(ZZ,1),[])';
% find correlations between the different elements...
ZZ = normalize_vec(ZZ)';
Z_ = ZZ*ZZ';
[v,iv] = sort(Z_,'descend');
figure,imagesc(Z_)
figure,imagesc(iv)
q =ie(1);
iv(1:5,q)
figure,imshow(jettify(Zs{q}))
figure,imshow(multiImage(jettify(Zs(iv(:,q))),false,true));
iv(2)
[A,AA] = visualizeClusters(conf_new,aa1,qq_1,'add_border',...
    true,'nDetsPerCluster',5,'gt_labels',true(size(aa1)),...
    'disp_model',true,'interactive',false);

imwrite(clusters2Images(A(ie)),'dest_ent.jpg');

% now we can check the consistency w.r.t each map of each image.
cMap = zeros(length(aa1),length(qq_1));

% get all detections for all images!
for k = 1:length(qq_1)
    curLocs = qq_1(k).cluster_locs;
    curCenters = round(boxCenters(curLocs));
    curZ = Zs{k};
    cMap(curLocs(:,11),k) = curZ(sub2ind(size(curZ),curCenters(:,2),curCenters(:,1)));
end

e1 = ents/max(ents);
e1 = (1-e1);
plot(e1);
new_grades = mean(cMap,2);
[n,in] = sort(new_grades,'descend');
[m,im] = sort(e1,'descend');
% e1(im(11:end)) = 0;
new_grades2 = cMap*e1(:);
[n2,in2] = sort(new_grades2,'descend');
figure,imshow(multiImage(aa1(1:10:400)))
% figure,imshow(multiImage(aa1(in(1:20:800))))
figure,imshow(multiImage(aa1(in2(1:10:400))))

figure,imshow(multiImage(aa1(gt_labels)));
figure,imshow(multiImage(aa1(in2(gt_labels(in2)))))

% cool, that worked nicely. now re-train the ones with high consistency from
% the top detections

[s,is] = sort(new_grades2,'descend');
gt_labels2 = cur_labels(cur_top_dets.cluster_locs(is,11));
[prec,rec,aps,T] = calc_aps2(cur_top_dets.cluster_locs(:,12),gt_labels);
[prec2,rec2,aps2,T2] = calc_aps2(new_grades2,gt_labels);
[prec3,rec3,aps3,T3] = calc_aps2(new_grades,gt_labels);
plot(cumsum(T))
hold on;
plot(cumsum(T2),'g');
plot(cumsum(T3),'r');

tt = 10;
for k = 1:tt
    k
    curDets = qq_1(ie(k)).cluster_locs;
    ids = aa1(curDets(1:20,11));
    conf_new.detection.params.detect_save_features = 1;
    suff = ['retrain_local_' num2str(k)];
    r = applyToSet(conf_new,qq_1(ie(k)),ids,[],[suff '_sample'],...
        'toSave',false,'nDetsPerCluster',50);
    conf_new.detection.params.detect_save_features = 0;
    conf_new.max_image_size = 100;
    conf_new.clustering.num_hard_mining_iters = 5;
    r_trained = train_patch_classifier(conf_new,r,nonPersonIds,'toSave',true,'suffix',...
        suff,'overRideSave',false);
    conf_new.max_image_size = conf.max_image_size;
    conf_new.detection.params.detect_save_features = 0;
    conf_new.detection.params.detect_min_scale = .5;
    rr = applyToSet(conf_new,r_trained,aa1(1:end),[],suff,'toSave',true,...
        'nDetsPerCluster',10,'disp_model',true,'override',false);
end

%% it seems the consistency reorders the true detections nicely. now I have to learn
%% a good classifier for such detections...

true_inds_orig = find(gt_labels);
true_inds_reordered = in2(gt_labels(in2));

% find the permutation between the original to reordered set...
[tf,orig_to_reordered] = ismember(true_inds_reordered,true_inds_orig);

figure,imshow(multiImage(aa1(true_inds_orig)));
figure,imshow(multiImage(aa1(true_inds_reordered),false,true));
% re-sample the original hogs from the images that created them. train on
% selected consistent faces.
ids_orig = train_ids(dets_top_test(1).cluster_locs(true_inds_orig,11));

% reordered is the same set. we'll have to train and then reorder!
% ids_reordered = train_ids(dets_top_test(1).cluster_locs(true_inds_reordered,11));

% make sure that this detector indeed creates the desired results...
conf.detection.params.detect_save_features = 1;
q_orig = applyToSet(conf,cur_top_dets,ids_orig,[],'ids_orig_check','override',false,...
    'nDetsPerCluster',inf);
conf.detection.params.detect_save_features = 0;
% q_orig;

% combine several classifiers...

newSels ={};
newSels{1} = [2 3 4 7 8];
newSels{2} = [1 16 17];
newSels{3} = [5 18 20 42];
newSels{4} = [6 21 22];
newSels{5} = [52 66 67];

q_orig_t = initClusters;
q_orig_t(1).isvalid = true;
for k =1:length(newSels)
    q_orig_t(k).cluster_locs = q_orig.cluster_locs(orig_to_reordered(newSels{k}),:);
    q_orig_t(k).cluster_samples = q_orig.cluster_samples(:,orig_to_reordered(newSels{k}));
    q_orig_t(k).isvalid = true;
    q_orig_t(k).w = q_orig_t(k).cluster_samples(:,1);
    q_orig_t(k).b = 0;
end

conf.clustering.num_hard_mining_iters = 12;
q_orig_t_d= train_patch_classifier(conf,q_orig_t,train_ids(~train_labels),'toSave',true,'suffix',...
    'retrain_consistent1_d','overRideSave',false);

[test_ids,test_labels] = getImageSet(conf,'test');
q_orig_t_d_test = applyToSet(conf,q_orig_t_d,test_ids,test_labels,...
    'ids_reorder_test_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true)
[prec,rec,aps,T,M]= calc_aps(q_orig_t_d_test,test_labels);
plot(rec,prec)

%%
q_orig_t_d_test2 = q_orig_t_d_test;
for k = 1:length(q_orig_t_d_test)
    q_orig_t_d_test2(k).cluster_locs(:,12) = sigmoid(q_orig_t_d_test2(k).cluster_locs(:,12)*ws(k)+bs(k));
%     M_s(:,k) = sigmoid(M(:,k)*ws(k)+bs(k));
end

q_orig_t_d_test_g = det_union(q_orig_t_d_test2([1 5]));
[prec_u,rec_u,aps_u,T_u,M_u]= calc_aps(q_orig_t_d_test_g,test_labels);
aps_u

%%


aps
[A,AA] = visualizeClusters(conf,test_ids,q_orig_t_d_test,'add_border',true,...
    'nDetsPerCluster',10,'height',64,'disp_model',true);
imwrite(clusters2Images(A),'ids_reorder_test_d12_1.jpg','Quality',100);

q_orig_t_d_train = applyToSet(conf,q_orig_t_d,train_ids,train_labels,...
    'ids_reorder_train_d12_1','nDetsPerCluster',10,'override',false,'disp_model',true);

a_train = visualizeLocs2(conf,train_ids,q_orig_t_d_train(1).cluster_locs(1:100,:),...
    'add_border',false);
figure,imshow(multiImage(a_train));

a_test = visualizeLocs2(conf,test_ids,q_orig_t_d_test(1).cluster_locs(1:100,:));

figure,imshow(multiImage(a_test));
imshow(a_test{1})

%% find mouth interactions.
[train_ids,train_labels] = getImageSet(conf,'train');

trues = train_labels(q_orig_t_d_train(1).cluster_locs(:,11));
a_train_true = a_train(trues(1:20));
rects = selectSamples(conf,a_train_true);

a_train_clean = visualizeLocs2(conf,train_ids,q_orig_t_d_train(1).cluster_locs(trues(1:20),:),...
    'add_border',false);

figure,imshow(multiImage(a_train_clean))

% sample gradients with finer details!
conf2 = conf;
conf2.detection.params.init_params.sbin = 4;
clusters = rects2clusters(conf2,rects,a_train_clean,[],1,0,false);
conf2_t = conf2;
conf2_t.max_image_size = 100;
% inflate a bit...
if (~exist('a_train_inf.mat','file'))
    a_train_inf = visualizeLocs2(conf,train_ids,q_orig_t_d_train(1).cluster_locs(:,:),...
        'add_border',false,'inflateFactor',1.5,'height',96,'saveMEmory',true);
    save a_train_inf  a_train_inf
else
    load a_train_inf;
end
negatives = a_train_inf(~trues(1:length(a_train_inf)));
positives = a_train_inf(trues(1:length(a_train_inf)));
clusts_smirc = train_patch_classifier(conf2_t,clusters,negatives,'toSave',true,'suffix',...
    'new_smircs1','overRideSave',false);

% testing - note, you can use the location here!!
clusts_smirc_train_check = applyToSet(conf2,clusts_smirc,a_train_inf,[],...
    'clusts_smirc_train_check','nDetsPerCluster',10,'override',false,'disp_model',true);

% testing - note, you can use the location here!!
clusts_smirc_train_check_pos = applyToSet(conf2,clusts_smirc,positives,[],...
    'clusts_smirc_train_check_pos','nDetsPerCluster',10,'override',false,'disp_model',true);
sz = [96 96];
Zs = createConsistencyMaps(clusts_smirc_train_check,sz,...
    find(trues(1:length(a_train_inf))),3);
figure,imagesc(Zs{1})
clusts_smirc_train_check_pos = applyToSet(conf2,clusts_smirc,positives,[],...
    'clusts_smirc_train_check_pos','nDetsPerCluster',10,'override',false,'disp_model',true,...
    'useLocation',Zs);

Zs_pos = createConsistencyMaps(clusts_smirc_train_check_pos,sz,...
    1:length(positives),1);
figure,imagesc(Zs_pos{3})


% now check on test...
if (~exist('a_test_inf.mat','file'))
    a_test_inf = visualizeLocs2(conf,test_ids,q_orig_t_d_test(1).cluster_locs,...
        'add_border',false,'inflateFactor',1.5,'height',96,'saveMemory',true);
    save a_test_inf a_test_inf;
else
    load a_test_inf
end

the_labels = test_labels(q_orig_t_d_test(1).cluster_locs(:,11));
[clusts_smirc_test_check,dets] = applyToSet(conf2,clusts_smirc,a_test_inf,the_labels,...
    'clusts_smirc_train_check','nDetsPerCluster',10,'override',false,'disp_model',true,...
    'useLocation',0,'add_border',true);

%%
Zs2 = Zs;
for k = 1:length(Zs)
    Zs2{k} = imfilter(Zs{k},fspecial('gaussian',25,9));
end
%%
Zs_pos = createConsistencyMaps(clusts_smirc_train_check_pos,sz,...
    1:length(positives),5);

% just remove really bad 

[clusts_smirc_test_check,dets,aps] = applyToSet(conf2,clusts_smirc,a_test_inf,the_labels,...
    'clusts_smirc_train_check','nDetsPerCluster',10,'override',false,'disp_model',true,...
    'useLocation',Zs_pos,'add_border',true,'dets',dets,'visualizeClusters',true);
%%
for theta =1
    re_graded = combine_grades(clusts_smirc_test_check,q_orig_t_d_test(1),theta,[]);
%     % re_graded = combine_grades(clusts_smirc_test_check(3),q_orig_t_d_test(1),theta,the_labels);
    [prec,rec,aps_,T,M] = calc_aps(re_graded,the_labels,sum(test_labels));
    [r,ir] = sort(aps_,'descend');
    disp(r(1:5));
    %
end

% %%
% [clusts_smirc_test_check_u] = det_union(re_graded(ir([1:2])));
% [prec,rec,aps_,T,M] = calc_aps(clusts_smirc_test_check_u,the_labels,sum(test_labels));
% plot(rec
% 
%%
plot(rec(:,ir(1)),prec(:,ir(1)))

[A,AA] = visualizeClusters(conf2,a_test_inf,re_graded(ir(1)),'add_border',...
    true,'nDetsPerCluster',10,'gt_labels',the_labels,...
    'disp_model',false,'interactive',false);

% figure,imshow(multiImage(AA))

imwrite(clusters2Images(A(ir)),'re_ordered.jpg');

%% ok, now try learning an even stronger clasiffier for the locals...
conf2_t.detection.params.detect_save_features = 1;
clusts_smirc_train_check_pos = applyToSet(conf2_t,clusts_smirc,positives,[],...
    'clusts_smirc_train_check_pos','nDetsPerCluster',10,'override',false,'disp_model',true);
%%
for k = 1:length(clusts_smirc_train_check_pos)
[A,AA] = visualizeClusters(conf2,positives,clusts_smirc_train_check_pos(k),'add_border',...
    false,'nDetsPerCluster',inf,'disp_model',false);

figure(1),imshow(multiImage(AA,false,true));title(num2str(k));
pause
end
%%

newSels_small ={};
newSels_small{1} = [1 2 5 17 19 27];
newSels_small{2} = [1 3 4 5 6 8 9];
newSels_small{3} = [1 2 60];
newSels_small{4} = [1 2 3 7 25 29];
newSels_small{5} = [1 2 3 5];
newSels_small{6} = [1];
newSels_small{7} = [1 2 5];
newSels_small{8} = [1 2];
newSels_small{9} = [1];
newSels_small{10} = [1 15 16 17];

new_train_small = initClusters;
new_train_small(1).isvalid = true;
for k =1:length(newSels_small)
    new_train_small(k).cluster_locs = clusts_smirc_train_check_pos(k).cluster_locs(newSels_small{k},:);
    new_train_small(k).cluster_samples = clusts_smirc_train_check_pos(k).cluster_samples(:,newSels_small{k});
    new_train_small(k).isvalid = true;
    new_train_small(k).w = new_train_small(k).cluster_samples(:,1);
    new_train_small(k).b = 0;
end

% m = visualizeLocs2(conf2,positives,new_train_small(1).cluster_locs);
% figure,imshow(multiImage(m))

conf.clustering.num_hard_mining_iters = 12;
maxImageSize = conf2.max_image_size;

conf2.max_image_size = 100;
% new_train_small= train_patch_classifier(conf2,new_train_small,train_ids(~train_labels),'toSave',true,'suffix',...
%     'new_train_small','overRideSave',true);
new_train_small= train_patch_classifier(conf2,new_train_small,nonPersonIds,'toSave',true,'suffix',...
    'new_train_small','overRideSave',false);
conf2.max_image_size=maxImageSize;
new_train_small_d = applyToSet(conf2,new_train_small,positives,[],...
    'new_train_small_train_check_pos_np','nDetsPerCluster',10,'override',false,'disp_model',true);

Zs_pos2 = createConsistencyMaps(new_train_small_d,sz,...
    1:length(positives),3);
imshow(multiImage(jettify(Zs_pos2)));
%%
[new_test_small_d,dets_small_d,aps_small_d] = applyToSet(conf2,new_train_small,a_test_inf,the_labels,...
    'new_train_small_test_pos_np','nDetsPerCluster',10,'override',false,'disp_model',true,...
    'useLocation',0,'add_border',true,'dets',dets_small_d,'visualizeClusters',true);

%%
for theta =.5
    re_graded = combine_grades(new_test_small_d,q_orig_t_d_test(1),theta,[]);
%     % re_graded = combine_grades(clusts_smirc_test_check(3),q_orig_t_d_test(1),theta,the_labels);
    [prec,rec,aps_,T,M] = calc_aps(re_graded,the_labels,sum(test_labels));
    [r,ir] = sort(aps_,'descend');
    disp(r(1:5));
    %
end

re_graded_u = det_union(re_graded(ir([1 3])));
[prec_u,rec_u,aps_u,T_u,M_u] = calc_aps(re_graded_u,the_labels,sum(test_labels));
aps_u
plot(rec_u,prec_u)

%%
%%
plot(rec,prec)
[A,AA] = visualizeClusters(conf2,a_test_inf,re_graded,'add_border',...
    true,'nDetsPerCluster',10,'gt_labels',the_labels,...
    'disp_model',false,'interactive',false);
% figure,imshow(multiImage(AA))
imwrite(clusters2Images(A),'re_ordered_small.jpg');

%% try again with a larger sbin, keep the rectangles....
conf3 = conf;
conf3.features.winsize =5;
conf3.detection.params.init_params.sbin = 8;
toShow = false;
new_train_small4 = initClusters;
new_train_small4(1).isvalid = true;
for q = 1:length(new_train_small)
    cc = rects2clusters(conf3,new_train_small(q).cluster_locs,...
        positives(new_train_small(q).cluster_locs(:,11)),1:length(positives),toShow);    
    new_train_small4(q).b = 0;
    new_train_small4(q).isvalid = true;
    for qq = 1:length(cc)
        new_train_small4(q).cluster_locs = [new_train_small4(q).cluster_locs;...
            cc(qq).cluster_locs];
        new_train_small4(q).cluster_samples = [new_train_small4(q).cluster_samples,...
            cc(qq).cluster_samples];
    end
    new_train_small4(q).w = new_train_small4(q).cluster_samples(:,1);        
end
%%
conf3.max_image_size = 100;
% new_train_small= train_patch_classifier(conf2,new_train_small,train_ids(~train_labels),'toSave',true,'suffix',...
%     'new_train_small','overRideSave',true);
new_train_small4= train_patch_classifier(conf3,new_train_small4,nonPersonIds,'toSave',true,'suffix',...
    'new_train_small4','overRideSave',true);
conf3.max_image_size = conf.max_image_size;

new_train_small4_check = applyToSet(conf3,new_train_small4,positives,[],...
    'new_train_small4_train_check_pos','nDetsPerCluster',10,'override',true,'disp_model',true);


[new_test_small4_check,dets,aps] = applyToSet(conf3,new_train_small4,a_test_inf,the_labels,...
    'new_test_small4_check','nDetsPerCluster',10,'override',true,'disp_model',true,...
    'useLocation',0,'add_border',true,'dets',dets,'visualizeClusters',true);
%%
for theta =1
    re_graded = combine_grades(new_test_small4_check,q_orig_t_d_test(1),theta,[]);
%     % re_graded = combine_grades(clusts_smirc_test_check(3),q_orig_t_d_test(1),theta,the_labels);
    [prec,rec,aps_,T,M] = calc_aps(re_graded,the_labels,sum(test_labels));
    [r,ir] = sort(aps_,'descend');
    disp(r(1:5));
    %
end

re_graded_u = det_union(re_graded(ir([1 2 3 6 7])));
[prec_u,rec_u,aps_u,T_u,M_u] = calc_aps(re_graded_u,the_labels,sum(test_labels));
aps_u
% plot(rec_u,prec_u)

%%
[A,AA] = visualizeClusters(conf3,a_test_inf,re_graded_u,'add_border',...
    true,'nDetsPerCluster',100,'gt_labels',the_labels,...
    'disp_model',false,'interactive',false);
figure,imshow(multiImage(AA))

%% good - now we can try to do this for a couple of detectors (one for side-bottle,
% one for holding-cup
a_train5 = visualizeLocs2(conf,train_ids,q_orig_t_d_train(5).cluster_locs(1:100,:),...
    'add_border',false);
figure,imshow(multiImage(a_train5,false,true));
newSels5 ={};
newSels5{1} = [1 2 3 4 18 34 54 55];
newSels5{2} = [8 12 77];

ii = q_orig_t_d_train(5).cluster_locs(1:100,11);
conf.detection.params.detect_save_features = 1;
q_orig_t_d_train2 = applyToSet(conf,q_orig_t_d_train(5),train_ids(ii),train_labels(ii),...
    'ids_reorder_train_d12_1_f','nDetsPerCluster',10,'override',true,'disp_model',true);
conf.detection.params.detect_save_features = 0;

q_orig_t5 = initClusters;
q_orig_t5(1).isvalid = true;
for k =1:length(newSels5)
    q_orig_t5(k).cluster_locs = q_orig_t_d_train2.cluster_locs((newSels5{k}),:);
    q_orig_t5(k).cluster_samples = q_orig_t_d_train2.cluster_samples(:,(newSels5{k}));
    q_orig_t5(k).isvalid = true;
    q_orig_t5(k).w = q_orig_t5(k).cluster_samples(:,1);
    q_orig_t5(k).b = 0;
end

conf.clustering.num_hard_mining_iters = 12;
q_orig_t5= train_patch_classifier(conf,q_orig_t5,train_ids(~train_labels),'toSave',true,'suffix',...
    'retrain_consistent1_d_5','overRideSave',true);

[test_ids,test_labels] = getImageSet(conf,'test');
q_orig_t_d_test5 = applyToSet(conf,q_orig_t5,test_ids,test_labels,...
    'ids_reorder_test_d12_1_5','nDetsPerCluster',10,'override',true,'disp_model',true)

q_orig_t_d_train5 = applyToSet(conf,q_orig_t5,train_ids,train_labels,...
    'ids_reorder_train_d12_1_5','nDetsPerCluster',10,'override',true,'disp_model',true);

[prec,rec,aps,T,M]= calc_aps(q_orig_t_d_test5,test_labels);
plot(rec,prec)

[q_orig_t_d_test5_u] = det_union(q_orig_t_d_test5([1 2]));

% make a logreg for this too
%%
n_to_check = 50;
n = length(q_orig_t_d_train5);
q_orig_t_d_test5_2 = q_orig_t_d_test5;

ws5 = zeros(size(q_orig_t_d_train5));
bs5 = zeros(size(q_orig_t_d_train5));
for k = 1:n
    X  = q_orig_t_d_train5(k).cluster_locs(1:n_to_check,12);
    y = train_labels(q_orig_t_d_train5(k).cluster_locs(1:n_to_check,11));
    [ws5(k),bs5(k)] = logReg(X, y);    
    q_orig_t_d_test5_2(k).cluster_locs(:,12) = ...
        sigmoid(q_orig_t_d_test5_2(k).cluster_locs(:,12)*ws5(k)+bs5(k));
end
% q_orig_t_d_train5
[q_orig_t_d_test5_u2] = det_union(q_orig_t_d_test5_2([1]));

[prec_,rec_,aps_,T_,M_]= calc_aps(q_orig_t_d_test5_u2,test_labels);
% now unite with the first one...
u2 = cat(1,q_orig_t_d_test5_u2,q_orig_t_d_test_g);
[u2] = det_union(u2);
[prec_,rec_,aps_,T_,M_]= calc_aps(u2,test_labels);
aps_
%%
% excellent! 25%
plot(rec_,prec_)
[A,AA] = visualizeClusters(conf2,test_ids,u2,'add_border',...
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

%% find non-detected drinking...
[r,ir] = sort(M_,'descend');
%figure,plot(ir)
t = test_labels(ir);
% show non-detected test images.
f = find(t);
sel_ = 50:100;
for k = 1:length(sel_)
    k
    imshow(getImage(conf,test_ids{ir(f(sel_(k)))}));
    pause;
end


%%
for theta = .5
    re_graded = combine_grades(clusts_smirc_test_check,u2_p,theta,[]);
%     % re_graded = combine_grades(clusts_smirc_test_check(3),q_orig_t_d_test(1),theta,the_labels);
    [prec,rec,aps_,T,M] = calc_aps(re_graded,the_labels,sum(test_labels));
    [r,ir] = sort(aps_,'descend');
    disp(r(1:5));
end