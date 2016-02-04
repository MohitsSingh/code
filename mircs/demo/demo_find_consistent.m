% clear classes;
initpath;
config;

%%
load dets_top_train_fixed;
ip = 13;
[test_ids,test_labels] = getImageSet(conf,'train',1,0);

[prec,rec,aps,T,M] = calc_aps(dets_top_train,test_labels);

t = test_labels(dets_top_train(ip(1)).cluster_locs(:,11));

aa_train = visualizeLocs2(conf,test_ids,dets_top_train(ip(1)).cluster_locs(t,:),'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false);
save aa_train13_inflate1 aa_train
% load aa_train13.mat
% imwrite(multiImage(aa_train),'a_train_true_inflate1.tif');

% now choose some random samples...
conf_new = conf;
conf_new.features.winsize = 4;
conf_new.detection.params.init_params.sbin = 4;
[X,locs] = samplePatches(conf_new,aa_train,1/3);

% train a classifier for each one...
%% 

clusts = makeClusters(X,locs);
% find negatives....
ids_neg = getNonPersonIds(VOCopts);
% aa_train_neg = visualizeLocs2(conf,test_ids,dets_top_train(ip(1)).cluster_locs(t_neg(1:100),:),'add_border',false,...
%     'inflateFactor',2,'height',128,'draw_rect',false);
conf_new.clustering.num_hard_mining_iters = 12;
clusts_trained = train_patch_classifier(conf_new,clusts,ids_neg,'toSave',true,'suffix',...
    'clusters_drinking_const13','overRideSave',true);

qq = applyToSet(conf_new,clusts_trained,aa_train,...
    true(size(aa_train)),'clusters_smirc13','toSave',true,'nDetsPerCluster',10,...
    'add_border',false,'override',false,'disp_model',true,'override',true);

Zs = {};
p = aa_train{1};
for k = 1:length(qq)    
    curZ = drawBoxes(zeros(size(p)),qq(k).cluster_locs,[],2);
    curZ = curZ/sum(curZ(:));
    Zs{k} = curZ;
%     imagesc(curZ); title(num2str(k)); pause;
end

% sort by entropy...
ents =zeros(size(Zs));
for  k = 1:length(Zs)
    curZ = Zs{k};        
    ents(k) = entropy(curZ/max(curZ(:)));
end

[e,ie] = sort(ents,'ascend');
[A,AA] = visualizeClusters(conf_new,aa_train,qq,'add_border',...
    true,'nDetsPerCluster',5,'gt_labels',true(size(aa_train)),...
    'disp_model',true);

%% retrain using the top-5 detections of the most consistent detectors...
clusts_trained2 = qq(ie(1));
conf_new.detection.params.detect_save_features = 1;
mwe= conf_new.detection.params.detect_max_windows_per_exemplar;
conf_new.detection.params.detect_max_windows_per_exemplar = 1;
qq2 = applyToSet(conf_new,clusts_trained2,aa_train,...
    col(true(size(aa_train))),'clusters_smirc13','toSave',false,'nDetsPerCluster',10,...
    'add_border',false,'override',false,'disp_model',false,'override',false);
conf_new.detection.params.detect_save_features = 0;
conf_new.detection.params.detect_max_windows_per_exemplar = mwe;
qq2.cluster_locs  =qq2.cluster_locs(1:10,:);
qq2.cluster_samples=qq2.cluster_samples(:,1:10);
clusts_trained = train_patch_classifier(conf_new,qq2,ids_neg,'toSave',true,'suffix',...
    'clusters_drinking_const13_min_ent','overRideSave',true);

conf_new.detection.params.detect_min_scale = .1;
conf_new.detection.params.detect_levels_per_octave = 5;
qq3= applyToSet(conf_new,clusts_trained,aa_train,...
    col(true(size(aa_train))),'clusters_smirc13_2','toSave',false,'nDetsPerCluster',100,...
    'add_border',false,'override',true,'disp_model',true,'override',true,'useLocation',Zs{ie(1)});

[A,AA] = visualizeClusters(conf_new,aa_train,qq3,'add_border',...
    true,'nDetsPerCluster',inf,'gt_labels',true(size(aa_train)),...
    'disp_model',true,'interactive',true)'

imwrite(clusters2Images(A(ie)),'smircs_ent.jpg');
big_test = visualizeLocs2(conf,test_ids,dets_top_train(ip(1)).cluster_locs,'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false,'saveMemory',true);
save big_test_13_inflate2 big_test

% load big_test
%% now apply the learned detectors inside the "big_test"
matlabpool
%%
[train_ids,train_labels] = getImageSet(conf,'train',1,0);

labels = train_labels(dets_top_train(ip(1)).cluster_locs(:,11));
qq_bigtest = applyToSet(conf_new,clusts_trained,big_test,...
    labels,'big_test_train_sub','toSave',true,'nDetsPerCluster',50,...
    'add_border',true,'useLocation',0);%Zs{ie(1)});

[a,b,c] = intersect(qq_bigtest(1).cluster_locs(:,11),find(labels));
r = qq_bigtest(1);
r.cluster_locs = r.cluster_locs(b,:);

[A,AA] = visualizeClusters(conf_new,big_test,qq_bigtest,'add_border',...
    true,'nDetsPerCluster',...
    inf,'gt_labels',labels,...
    'disp_model',true,'interactive',false);

imwrite(multiImage(AA(1:225)),'mmm.jpg');

imwrite(clusters2Images(A(ie)),'big_test_train_sub_ent.jpg');


imshow(multiImage(AA))

qq_bu = qq_bigtest;
%%

% for j = 300
qq_bigtest = qq_bu;
for k = 1:length(qq_bigtest)
    curLocs = qq_bigtest(k).cluster_locs;
    curScore = curLocs(:,12);
    
        curLocs(:,12) = .01*rand(size(curScore))+ 2*curScore +dets_top_train(ip(1)).cluster_locs(curLocs(:,11),12);
%     curLocs(:,12) = .01*rand(size(curScore))+ w(1)*curScore +w(2)*dets_top_train(ip(1)).cluster_locs(curLocs(:,11),12);
    [s,is] = sort(curLocs(:,12),'descend');
    curLocs = curLocs(is,:);
    qq_bigtest(k).cluster_locs = curLocs;
end


[prec,rec,aps,T,M] = calc_aps(qq_bigtest,labels);
disp(sort(aps,'descend'));
% end
[a,ia] = sort(aps,'descend');
%%
[A,AA] = visualizeClusters(conf_new,big_test,qq_bigtest,'add_border',...
    true,'nDetsPerCluster',50,'gt_labels',labels);
imwrite([clusters2Images(A(ia))],['local_test_trunc.jpg']);

plot(rec(:,ia(1)),prec(:,ia(1)))
%%

bb = visualizeLocs2(conf_new,big_test,qq_bigtest(1).cluster_locs(1:100,:),'add_border',false,...
    'inflateFactor',2,'height',128,'draw_rect',true,'saveMemory',true);
imwrite(AA{1},'a1.tif');

%% make a verifier, or something to learn non-faces....
% pos_eyes = selectSamples(conf,aa_train(1:5));
% save left_eye_rects pos_eyes
load left_eye_rects
conf_new2 = conf_new;
conf_new2.detection.params.init_params.sbin = 4;
clusts_eye = rects2clusters(conf_new2,pos_eyes,aa_train(1:5),1:5,true);
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
ids = ids(t==-1); % persons non gratis
conf_new2.clustering.num_hard_mining_iters = 12;
eye_clusts_trained = train_patch_classifier(conf_new2,clusts_eye,ids,'toSave',true,'suffix',...
    'eyeclassifiers_sbin4');
for k = 1:length(eye_clusts_trained)
    eye_clusts_trained(k).refLocation = boxCenters(eye_clusts_trained(k).cluster_locs);
end

%% learn right eyes too!
% pos_eyes_right = selectSamples(conf,aa_train(1:5));
% save right_eye_rects pos_eyes_right
load right_eye_rects;
conf_new2 = conf_new;
conf_new2.detection.params.init_params.sbin = 4;
clusts_eye_right = rects2clusters(conf_new2,pos_eyes_right,aa_train(1:5),1:5,true);
cls = 15; % class of person;
[ids,t] = textread(sprintf(VOCopts.imgsetpath,[VOCopts.classes{cls} '_train']),'%s %d');
ids = ids(t==-1); % persons non gratis
right_eye_clusts_trained = train_patch_classifier(conf_new2,clusts_eye_right,ids,'toSave',true,'suffix',...
    'righteyeclassifiers_sbin4');
for k = 1:length(eye_clusts_trained)
    right_eye_clusts_trained(k).refLocation = boxCenters(right_eye_clusts_trained(k).cluster_locs);
end

qq_eyes_learn_right = applyToSet(conf_new2,right_eye_clusts_trained,small_test,...
    true(size(small_test)),'righteye_to_learn','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',0);

p = small_test{1};
Z_right = zeros(size(p,1));
for k = 1:length(qq_eyes_learn)    
    Z_right = Z_right + drawBoxes(Z_right,qq_eyes_learn_right(k).cluster_locs,[],2);
end
Z_right = Z_right/max(Z_right(:));
figure,imagesc(Z_right)


%% accummulate eye locations!
qq_eyes_learn = applyToSet(conf_new2,eye_clusts_trained,small_test,...
    true(size(small_test)),'eye_to_learn','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',0);

p = small_test{1};
Z = zeros(size(p,1));
for k = 1:length(qq_eyes_learn)    
    Z = Z + drawBoxes(Z,qq_eyes_learn(k).cluster_locs,[],2);
end
Z = Z/max(Z(:));

qq_eyes4 = applyToSet(conf_new2,eye_clusts_trained,big_test,...
    labels,'eye_test_big4','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',Z);

qq_righteyes4 = applyToSet(conf_new2,right_eye_clusts_trained,big_test,...
    labels,'righteye_test_big4','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',Z_right);

%% % now let's try to combine all scores 
% first, find the eye score for each image.
curLocs = qq_eyes4(1).cluster_locs;
curLocs_right = qq_righteyes4(1).cluster_locs;
eye_penalty = 0;
eye_scores_p = -eye_penalty*ones(size(big_test));
eye_scores_p(curLocs(:,11)) = curLocs(:,12);
eye_scores_p_right = -eye_penalty*ones(size(big_test));
eye_scores_p_right(curLocs_right(:,11)) = curLocs_right(:,12);

qq_bigtest = qq_bu;
ws = zeros(4,size(qq_bigtest));
bs = zeros(size(qq_bigtest));
for k = 1:length(qq_bigtest)
    curLocs = qq_bigtest(k).cluster_locs;
    curScore = curLocs(:,12);
%                 
%     X = [.01*rand(size(curScore))+ curScore,dets_top_train(ip(1)).cluster_locs(curLocs(:,11),12),...
%         double(col(eye_scores_p(curLocs(:,11)))),...
%         double(col(eye_scores_p_right(curLocs(:,11))))];
%     y = labels(curLocs(:,11));
%     [w b ap] = checkSVM( X,y==1);
%     ws(:,k) = w;
%     bs(k) = b;
    curScore = rand(size(curScore))*.001+...
        curScore+ .1*dets_top_train(ip(1)).cluster_locs(curLocs(:,11),12) +...
       +.2*((double(col(eye_scores_p(curLocs(:,11))))+...
        +double(col(eye_scores_p_right(curLocs(:,11))))));
%     curScore = X*w; 
    qq_bigtest(k).cluster_locs(:,12) = curScore;
    [s,is] = sort(curScore,'descend');   
    qq_bigtest(k).cluster_locs = qq_bigtest(k).cluster_locs(is,:);
end
[prec,rec,aps,T,M] = calc_aps(qq_bigtest,labels);
disp(sort(aps,'descend'));
% end
[a,ia] = sort(aps,'descend');

%%
[A,AA] = visualizeClusters(conf_new,big_test,qq_bigtest,'add_border',...
    true,'nDetsPerCluster',50,'gt_labels',labels);
imwrite([clusters2Images(A(ia))],['local_test_trunc.jpg']);
plot(rec(:,ia(1)),prec(:,ia(1)));
%%
bb = visualizeLocs2(conf_new,big_test,qq_bigtest(1).cluster_locs(1:100,:),'add_border',false,...
    'inflateFactor',2,'height',128,'draw_rect',true,'saveMemory',true);

bb = visualizeLocs2(conf_new2,big_test,qq_eyes4(1).cluster_locs(1:500,:),'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false,'saveMemory',false);

imwrite(multiImage(bb),'eyes_big.jpg');

% imwrite(AA{1},'a1.tif');
%% and another small experiment - sum all eye locs...(used above...)
p = small_test{1};
Z_all = zeros(size(p,1));
for k = 1:length(qq_eyes4)    
   Z_all = Z_all + drawBoxes(Z_all,qq_eyes_learn(k).cluster_locs,[],2);
end
Z_all = Z_all/max(Z_all(:));
% interestingly, it has a pereference for left eyes.


%% ok, now use on test!
load dets_top_test
[test_ids,test_labels] = getImageSet(conf,'test');
big_test_test = visualizeLocs2(conf,test_ids,dets_top_test(13).cluster_locs,'add_border',false,...
    'inflateFactor',2,'height',128,'draw_rect',false,'saveMemory',true);
labels = test_labels(dets_top_test(13).cluster_locs(:,11));
% save big_test_test big_test_test

qq_eyes4 = applyToSet(conf_new2,eye_clusts_trained,big_test_test,...
    labels,'eye_test_big4_test','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',Z);

bb = visualizeLocs2(conf_new2,big_test_test,qq_eyes4(1).cluster_locs(1:500,:),'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false,'saveMemory',false);

imwrite(multiImage(bb),'eyes_big_test.jpg');

qq_righteyes4 = applyToSet(conf_new2,right_eye_clusts_trained,big_test_test,...
    labels,'righteye_test_big4_test','toSave',true,'nDetsPerCluster',50,...
    'add_border',false,'useLocation',Z_right);

bb = visualizeLocs2(conf_new2,big_test_test,qq_righteyes4(1).cluster_locs(1:500,:),'add_border',false,...
    'inflateFactor',1,'height',64,'draw_rect',false,'saveMemory',false);
imwrite(multiImage(bb),'right_eyes_big_test.jpg');

qq_bigtest_test = applyToSet(conf_new,clusts_trained,big_test_test,...
    labels,'local_big_test_1_test','toSave',true,'nDetsPerCluster',50,...
    'add_border',true,'useLocation',ones(sizeZ_c));
qq_bu = qq_bigtest_test;
%%
qq_bigtest_test = qq_bu;
curLocs = qq_eyes4(1).cluster_locs;
curLocs_right = qq_righteyes4(1).cluster_locs;
eye_penalty = 0;
eye_scores_p = -eye_penalty*ones(size(big_test_test));
eye_scores_p(curLocs(:,11)) = curLocs(:,12);
eye_scores_p_right = -eye_penalty*ones(size(big_test_test));
eye_scores_p_right(curLocs_right(:,11)) = curLocs_right(:,12);

for k = 1:length(qq_bigtest_test)
    curLocs = qq_bigtest_test(k).cluster_locs;
    curScore = curLocs(:,12);
    curScore = rand(size(curScore))*.001+...
        curScore+ dets_top_test(ip(1)).cluster_locs(curLocs(:,11),12) +...
       +.2*(double(col(eye_scores_p(curLocs(:,11))))+...
        +double(col(eye_scores_p_right(curLocs(:,11)))))
%     X = [.01*rand(size(curScore))+ curScore,dets_top_test(ip(1)).cluster_locs(curLocs(:,11),12),...
%         double(col(eye_scores_p(curLocs(:,11)))),...
%         double(col(eye_scores_p_right(curLocs(:,11))))];
%     
%     curScore = X*ws(:,k);
   
    qq_bigtest_test(k).cluster_locs(:,12) = curScore;
    [s,is] = sort(curScore,'descend');   
    qq_bigtest_test(k).cluster_locs = qq_bigtest_test(k).cluster_locs(is,:);
end
[prec,rec,aps,T,M] = calc_aps(qq_bigtest_test,labels);
disp(sort(aps,'descend'));
% end
[a,ia] = sort(aps,'descend');
%%
%%
[A,AA] = visualizeClusters(conf_new,big_test_test,qq_bigtest_test,'add_border',...
    true,'nDetsPerCluster',50,'gt_labels',labels);
imwrite([clusters2Images(A(ia))],['local_test_trunc_test.jpg']);
plot(rec(:,ia(1)),prec(:,ia(1)))

tt = test_labels(dets_top_test(ip(1)).cluster_locs(:,11));
aa_test = visualizeLocs2(conf,test_ids,dets_top_test(ip(1)).cluster_locs(tt,:),'add_border',false,...
    'inflateFactor',[1 2],'height',64,'draw_rect',false);
imwrite(multiImage(aa_test),'a_test_true_1.tif');