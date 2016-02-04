initpath;
config;
load lipData.mat;
close all;

initLandMarkData;

% lipImages_train = lipImages_train2;
% lipImages_test = lipImages_test2;
% 
posLipImages= lipImages_train(t_train);
posInds = find(t_train);
ignoreList = false(size(posLipImages));
% figure,imshow(multiImage(posLipImages));

% drinking from straw
f = find(~ignoreList); 
strawInds = [1 5 6 14 18 19 21 23 27 31 38 42 46 51 54];
ignoreList (strawInds) = true;
strawInds_abs = posInds(strawInds);
posInds(ignoreList) = [];
makeSpecializedFunction('straw');

figure,imshow(multiImage(posLipImages(f(strawInds))))
mImage(lipImages_train(strawInds_abs));
% what is the sub-pose? 
face_comp(strawInds_abs)
    

strawSubImages = posLipImages(f(strawInds));

f = find(~ignoreList); 
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 7 8 18 21 27 36 37 43 44 45 47];
cupInds_1 = f(m);
cupInds_1_abs = posInds(m);
ignoreList(cupInds_1) = true;
posInds(m) = [];
makeSpecializedFunction('cup_1'); % bottom cup
figure,imshow(multiImage(posLipImages(cupInds_1)))
cup1SubImages = posLipImages(cupInds_1);
mImage(lipImages_train(cupInds_1_abs));
face_comp(cupInds_1_abs)

f = find(~ignoreList);
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 6 11 13 22 25 26];
cupInds_2 = f(m);
cupInds_2_abs = posInds(m);
ignoreList(cupInds_2) = true;
posInds(m) = [];
makeSpecializedFunction('cup_2'); % side cup/ can with hand
figure,imshow(multiImage(posLipImages(cupInds_2)))
mImage(lipImages_train(cupInds_2_abs));
cup2SubImages = posLipImages(cupInds_2);
face_comp(cupInds_2)


f = find(~ignoreList); figure,imshow(multiImage(posLipImages(~ignoreList)));
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [3 5 6 12 13 15 20 24 26 27 28];
cupInds_3 = f(m);
cupInds_3_abs = posInds(m);
ignoreList(cupInds_3) = true;
posInds(m) = [];
makeSpecializedFunction('cup_3'); % side cup/ can with hand
figure,imshow(multiImage(posLipImages(cupInds_3)))
mImage(lipImages_train(cupInds_3_abs));
cup3SubImages = posLipImages(cupInds_3);
face_comp(cupInds_3)

f = find(~ignoreList);
% figure,imshow(multiImage(posLipImages(~ignoreList)));
m = [7 10 17];
bottleInds = f(m);
bottleInds_abs = posInds(m);
ignoreList(bottleInds) = true;
posInds(m) = [];
makeSpecializedFunction('bottle'); % side bottle/ can with hand
figure,imshow(multiImage(posLipImages(bottleInds)))
mImage(lipImages_train(bottleInds_abs));
bottleSubImages = posLipImages(bottleInds);
face_comp(bottleInds)

conf.features.vlfeat.cellsize = 8;
conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
conf.clustering.num_hard_mining_iters = 12;
conf.detection.params.detect_keep_threshold = -1;

close all;

%%

lipImages_train_100 = multiCrop(conf,lipImages_train,[],[100 100]);
lipImages_test_100 = multiCrop(conf,lipImages_test,[],[100 100]);

% [feats_train,feats_test] = extractBowFeats(lipImages_train_100,lipImages_test_100,'lipFeats.mat');
% [feats_train_c,feats_test_c] = extractColorFeats(lipImages_train_100,lipImages_test_100,'lipFeats_color.mat');

%dict = learnBowDictionary(conf,set1,true);
% dict = codebook;
% save('kmeans_4000.mat','dict');
load('kmeans_4000.mat');
model.vocab=  dict; clear dict;
model.numSpatialX = [1 2];
model.numSpatialY = [1 2];
model.kdtree = vl_kdtreebuild(model.vocab) ;
model.quantizer = 'kdtree';
% model.vocab = dict;
model.w = [] ;
model.b = [] ;
model.phowOpts = {'Step',1};

descs_train = getAllDescs(conf,model,get_full_image,[],'~/storage/train2_descs_4000.mat');
descs_test = getAllDescs(conf,model,test_faces_2,[],'~/storage/test2_descs_4000.mat');


close all;
mImage(strawSubImages);
override= false;

bundle.train_imgs = lipImages_train_100;
bundle.test_imgs = lipImages_test_100;
bundle.t_train = t_train;
bundle.t_test = t_test;
bundle.train_imgs_large = get_full_image;
bundle.test_imgs_large = test_faces_2;
bundle.roi_train = lipBoxes_train_r_2;
bundle.roi_test = lipBoxes_test_r_2;
bundle.model = model;
bundle.descs_train = descs_train;
bundle.descs_test = descs_test;

override = false;
model.numSpatialX = [2];
model.numSpatialY = [2];

% conf.detection.params.detect_min_scale= .8;
override = false;
[cup1Detector,q_train,q_test_cup1,conf3,svm_scores_cup1] = makeSpecializedDetector(conf,cup1SubImages,'cup1',override,...
    cupInds_1_abs,bundle);
[cup2Detector,q_train,q_test_cup2,conf3,svm_scores_cup2] = makeSpecializedDetector(conf,cup2SubImages,'cup2',override,...
    cupInds_2_abs,bundle);
[cup3Detector,q_train,q_test_cup3,conf3,svm_scores_cup3] = makeSpecializedDetector(conf,cup3SubImages,'cup3',override,...
    cupInds_3_abs,bundle);
[bottleDetector,q_train,q_test_bottle,conf3,svm_scores_bottle] = makeSpecializedDetector(conf,bottleSubImages,'bottle',override,...
    bottleInds_abs,bundle);
[strawDetector,q_train,q_test_straw,conf3,svm_scores_straw] = makeSpecializedDetector(conf,strawSubImages,'straw',override,...
    strawInds_abs,bundle);


%%
[strawDetector,q_train,q_test_straw,conf3,svm_scores_straw] = makeSpecializedDetector(conf,strawSubImages,'straw1',override,...
    strawInds_abs,bundle);
%%


X_neg = imageSetFeatures2(conf,lipImages_train_100(~t_train),true,[64 64]);
X_test = imageSetFeatures2(conf,lipImages_test_100,true,[64 64]);
%%
mImage(strawSubImages);
%%
close all;
X_pos = imageSetFeatures2(conf,strawSubImages([1]),true,[56 56]);
conf.features.winsize = [7 7 31];
% [ws,b,sv,coeff,svm_model] = train_classifier(X_pos,X_neg,10,1);
% figure,imshow(showHOG(conf,ws.^2));
% f = ws'*X_test;
% showSorted(lipImages_test_100,f,50);


%%
conf.max_image_size = 256;
clusters = makeCluster(X_pos,[]);
% train against non-person images.
conf.clustering.num_hard_mining_iters = 5;
c_0 = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'toSave',false);
figure,imshow(showHOG(conf,c_0))

conf.detection.params.detect_min_scale = .5;
conf.detection.params.detect_max_scale = 1;
q_train_face = applyToSet(conf,c_0,lipImages_train_100(t_train),[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

r_train_face_true = visualizeLocs2_new(conf,lipImages_train_100(t_train),q_train_face.cluster_locs);
mImage(r_train_face_true);


load ~/storage/train_gpbs.mat;


q_train_face_false = applyToSet(conf,c_0,train_faces(~t_train),[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

other_straw_images = multiRead(conf,'~/data/drinking_extended/straw','.jpg');
conf.detection.params.detect_min_scale = .5;
q_other= applyToSet(conf,c_0,other_straw_images,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

r_other = visualizeLocs2_new(conf,other_straw_images,q_other.cluster_locs);
mImage(r_other);
figure,imshow(multiImage(r_other));

d = 10;
x = imageSetFeatures2(conf,[r_faces(1:2) r_other([1:10 12:15 18])],true,[8*d 8*d]);
conf.features.winsize = [d d 31];

xc = makeCluster(x,[]);
conf.clustering.num_hard_mining_iters = 10;
c_1 = train_patch_classifier(conf,xc,lipImages_train_100(~t_train),'toSave',false,'keepSV',false,'C',.01);
figure,imshow(showHOG(conf,c_1));

% do training with cross-validation

q_test_face_true = applyToSet(conf,c_1,lipImages_test_100(t_test),[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);
r_faces_test_true = visualizeLocs2_new(conf,lipImages_test_100(t_test),q_test_face_true.cluster_locs);
mImage(r_faces_test_true);

% 
q_test_face = applyToSet(conf,c_1,lipImages_test_100,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);
r_faces_test = visualizeLocs2_new(conf,lipImages_test_100,q_test_face.cluster_locs);
indices = q_test_face.cluster_locs(:,11);
scores = q_test_face.cluster_locs(:,12);
a1 = mImage(lipImages_test_100(indices(1:50)));
a2 = mImage(r_faces_test(1:50));

a1 = imresize(a1,[size(a2,1),NaN]);
figure,imshow(a1);title('lip images  100');
figure,imshow(a2);title('detected straws');

X_r_test_true = imageSetFeatures2(conf,r_faces_test_true,true,[64 64]);
[~,~,svm_score] = svmpredict(zeros(size(X_r_test_true,2),1),X_r_test_true',svm_model);
showSorted(r_faces_test_true,svm_score);

q_test_face2 = applyToSet(conf,c_0,r_faces_test_true,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

r_faces_test_true1 = visualizeLocs2_new(conf,r_faces_test_true,q_test_face2.cluster_locs);

a = mImage(r_faces_test_true1);



q_test_face_0 = applyToSet(conf,c_0,test_faces,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

rr_features = imageSetFeatures2(conf,r_faces_test_0,true,[64 64]);
[~,~,svm_score_test] = svmpredict(zeros(size(rr_features,2),1),rr_features',svm_model);
showSorted(r_faces_test_0,svm_score_test,50);

r_faces_test_0 = visualizeLocs2_new(conf,test_faces,q_test_face_0.cluster_locs);
mImage(r_faces_test_0(1:250));

q_test_face_1 = applyToSet(conf,c_1,r_faces_test_0,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);


q = c_1.w'*rr_features;
%%
showSorted(r_faces_test_0,q+10*(q_test_face_0.cluster_locs(:,12)'),50);
%%

r_faces_test_1 = visualizeLocs2_new(conf,r_faces_test_0,q_test_face_1.cluster_locs);

mImage(r_faces_test_1(1:50));

% q0 = arrangeDet(q_test_face_0,'index');
q1 = arrangeDet(q_test_face_1,'index');

% figure,plot(q0.cluster_locs(:,12))
figure,plot(q1.cluster_locs(:,12))


%%
%showSorted(lipImages_test_100,f_test_0,50);
showSorted(lipImages_test_100,1*double(f_test_1)+100*double(f_test_0>-.9),50);

%%
[~,i_f_test_0] = sort(f_test_0,'descend');

% mImage(lipImages_test_100(i_f_test_0(t_test(i_f_test_0))));
%  mImage(lipImages_test_100(t_test));
figure,plot((i_f_test_0));

%figure,imshow(showHOG(conf3,strawDetector));

q_test_straw = arrangeDet(q_test_straw,'index');
mImage(lipImages_test_100(q_test_straw.cluster_locs(1:50,11)));
straw_scores = q_test_straw.cluster_locs(:,12);

trainDPM

%%
f = straw_scores+25*double(f_test_0)';
showSorted(lipImages_test_100,f,50);
%%


other_straw_images = multiRead('~/data/drinking_extended/straw','.jpg');
mImage(r);
conf.detection.params.detect_min_scale = .1;
q_test_face = applyToSet(conf,test_dets,other_straw_images,[],...
    [ 'nos_t_face'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);
r_faces = visualizeLocs2_new(conf,other_straw_images,q_test_face.cluster_locs);
mImage(r_faces);

conf3.detection.params.detect_max_scale = 2;
conf3.detection.params.detect_min_scale = .1;
q_test = applyToSet(conf3,strawDetector,r_faces,[],...
    [ 'nos_t'],'useLocation',0,'disp_model',true,'override',override,'toSave',false);

r = visualizeLocs2_new(conf3,r_faces,q_test.cluster_locs);
E = mImage(r);
mImage(r_faces(q_test.cluster_locs(:,11)));
E = mImage(r);
E = edge(imresize(im2double(rgb2gray(mImage(r))),2),'canny');
figure,imagesc(E)
figure,imagesc(mImage(r));


% add another svm score for the entire face region.

model2 = model;
model2.numSpatialX = [2 4];
model2.numSpatialY = [2 4];

feats_train = getBOWFeatures(conf,model2,get_full_image,[],descs_train);
feats_test = getBOWFeatures(conf,model2,test_faces_2,[],descs_test);

[~,~,~,~,svm_model] = train_classifier(feats_train(:,t_train),feats_train(:,~t_train));
[~,~,svm_scores] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model);


%%
d = 1;
cup1_scores = q_test_cup1.cluster_locs(:,12) + d*svm_scores_cup1;
cup2_scores = q_test_cup2.cluster_locs(:,12) + d*svm_scores_cup2;
cup3_scores = q_test_cup3.cluster_locs(:,12) + d*svm_scores_cup3;
bottle_scores = q_test_bottle.cluster_locs(:,12) + d*svm_scores_bottle;
straw_scores = q_test_straw.cluster_locs(:,12) + d*svm_scores_straw;

allScores = [cup1_scores,cup2_scores,cup3_scores,bottle_scores,straw_scores];
m = max(allScores,[],2)+0.5*svm_scores+10*double(test_face_scores(:)>-.81);
% m = max(allScores,[],2)+10*double(test_faces_scores_r(:)>-.81);
% m = straw_scores;
% m = sum(allScores,2);


[prec,rec,aps] = calc_aps2(allScores,t_test,sum(test_labels))

plot(rec,prec);
[prec,rec,aps] = calc_aps2(m,t_test,sum(test_labels))
hold on;
plot(rec,prec,'k--','LineWidth',2);
legend('cup1','cup2','cup3','bottle','straw','max')
%%
showSorted(test_faces_2,m,50);


% override = false;
% [cup3Detector,q_train,q_test_cup3,conf3] = makeSpecializedDetector(conf,cup3SubImages,'cup3',lipImages_train_100,lipImages_test_100,...
%     t_train,t_test,override,cupInds_3_abs);
% [prec,rec,aps] = calc_aps(q_test_cup3,t_test);

[bottleDetector,q_train,q_test_bottle,conf3] = makeSpecializedDetector(conf,bottleSubImages,'bottle',lipImages_train_100,lipImages_test_100,...
    t_train,t_test,override,bottleInds_abs);


all_res = [q_test_straw,q_test_cup1,q_test_cup2,q_test_cup3,q_test_bottle];
[prec,rec,aps] = calc_aps(all_res,t_test);
all_res_u = combineDetections(all_res);
[prec,rec,aps] = calc_aps(all_res_u,t_test);
plot(rec,prec)

[r,ir] = sort(all_res_u.cluster_locs(:,12),'descend');
mImage(lipImages_test(ir(1:50)));

% run this 

dets = [q_test_straw,q_test_cup1,q_test_cup2,q_test_cup3,q_test_bottle];

% run the cup 3 detector on all images....
q_train_cup3 = applyToSet(conf3,cup3Detector,test_faces,[],...
    'cup3_check1','useLocation',0,'disp_model',true,'override',false);

q_train_cup3 = applyToSet(conf3,cup3Detector,test_ids,[],...
    'cup3_check2','useLocation',0,'disp_model',true,'override',false,'nDetsPerCluster',...
    100);

% [prec,rec,aps] = calc_aps(q_train_cup3,test_labels);
% plot(rec,prec)



%%
dets_new = combineDetections(dets);
[prec,rec,aps] = calc_aps(dets_new,t_test);
 showSorted(lipImages_test,dets_new.cluster_locs(:,12),50)
%%

[prec,rec,aps] = calc_aps(q_test,t_test,sum(test_labels))
plot(rec,prec)


q_test = arrangeDet(q_test,'index');

m_test = multiCrop(conf,lipImages_test_100,round(q_test.cluster_locs),[50 50]);
% bowFeats_test_2 = getBOWFeatures(conf,model,m_test,[],[]);

q_scores = q_test.cluster_locs(:,12);

[~,~,~,~,svm_model] = train_classifier(feats_train(:,cupInds_3_abs),feats_train(:,~t_train));
[~,~,~,~,svm_model_h] = train_classifier(x_train(:,cupInds_3_abs),x_train(:,~t_train),[],[],2);
[~,~,svm_scores] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model);
[~,~,svm_scores_crop] = svmpredict(zeros(size(bowFeats_test_2,2),1),double(bowFeats_test_2'),svm_model);

[~,~,~,~,svm_model_color] = train_classifier(feats_train_c(:,cupInds_3_abs),feats_train_c(:,~t_train));
[~,~,svm_scores_color] = svmpredict(zeros(size(feats_test_c,2),1),double(feats_test_c'),svm_model);
[~,~,svm_scores_h] = svmpredict(zeros(size(x_test,2),1),double(x_test'),svm_model_h);


svm_scores1 = normalise(svm_scores);
svm_scores2 = normalise(svm_scores_color);
svm_scores3 = normalise(svm_scores_crop);
svm_scores4 = normalise(svm_scores_h);

% scores_t = q_scores+svm_scores1;
%%
t_score = 1*q_scores(:)+1*svm_scores1(:)+0*double(test_faces_scores_r(:) > -.5)+...
    1*svm_scores4(:);
%     3*svm_scores2(:);
%10*double(kp_score(:)>3);
[prec,rec,aps]  = calc_aps2(t_score,t_test)%,sum(test_labels))
[r,ir] = sort(t_score,'descend');

figure,plot(rec,prec); xlabel('recall'); ylabel('precision')
%%
% 
% %t_score
% ttt = t_score > 0;
% m = paintRule(test_faces,t_score>1.360,[],[],5);
% mm = m(t_test);
% mImage(mm);
% 
 m = paintRule(lipImages_test,t_test,[],[],3);
mImage(m(ir(1:50)));
% mImage(test_faces(ir(1:50)));
 
%  figure,plot(rec,prec)
 
%  figure,imshow(multiImage(test_ids(
    
%%


% q_test = arrangeDet(q_test,'index');
% test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);
% 
% test_faces = test_faces(test_face_scores>=min_test_score);

train_res = visualizeLocs2_new(conf3,lipImages_train_100(t_train),q_train.cluster_locs);
mImage(train_res);

[detector,q_train,q_test,conf3] = makeSpecializedDetector(conf,train_res([1:9]),'cup3_ref',lipImages_train_100,lipImages_test_100,...
    t_train,t_test,override);

train_res = visualizeLocs2_new(conf3,lipImages_train_100(t_train),q_train.cluster_locs);
mImage(train_res);

[detector,q_train,q_test,conf3] = makeSpecializedDetector(conf,train_res([1:9]),'cup3_ref',lipImages_train_100,lipImages_test_100,...
    t_train,t_test,override);

[detector_1,q_train,q_test,conf3] = makeSpecializedDetector(conf,train_res([1:9]),'cup3_ref_na',lipImages_train_100(non_action_train),lipImages_test_100,...
    t_train(non_action_train),t_test,false);

q_train_1 = applyToSet(conf3,detector,lipImages_train_100,[],...
    'cup3_ref_na_train','useLocation',0,'disp_model',true,'override',true);
 mmm = visualizeLocs2_new(conf3,lipImages_train_100,q_train_1.cluster_locs);
 mImage(mmm(1:50));
 
 q_train_1 = arrangeDet(q_train_1,'index');
  q_test = arrangeDet(q_test,'index');

 [~,~,~,~,svm_model] = train_classifier(feats_train(:,t_train),feats_train(:,~t_train)); 
 [~,~,svm_scores] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model);
 [r,ir] = sort(svm_scores,'descend');
 
r_scores = q_test.cluster_locs(:,12);
%%

  [r,ir] = sort(0*svm_scores+10*r_scores,'descend');
 %mImage(lipImages_test(ir(1:50)));
 mImage(test_faces(ir(1:50)));
 %%
%%
q_test_scores = q_test.cluster_locs(:,12)+.1*(x_1_test_scores);
[r,ir] = sort(q_test_scores,'descend');
mImage(test_faces(ir(1:50)));
q_test2 = q_test;
q_test2.cluster_locs(:,12) = q_test_scores;
[prec,rec,aps] = calc_aps(q_test2,t_test)


%%
train_res = visualizeLocs2_new(conf3,lipImages_train_100(t_train),q_train.cluster_locs);
mImage(train_res);

%%

%%

train_res = visualizeLocs2_new(conf3,lipImages_train_100(t_train),q_train.cluster_locs);
mImage(train_res);
[detector,q_train,q_test,conf3] = makeSpecializedDetector(conf,train_res(1:3),'bottle_ref',lipImages_train_100,lipImages_test_100,...
    t_train,t_test,override);



%%
% % close all;
% % cd  specialized/
% % addpath('specialized');
% % 
% % % find points of high curvature.
% % 
% % % in cup_1: Todo: fit elliptic curves to the edge of the cup.
% % % find if there is a solid segment "below" the curve, and that above it
% % % there are the eyes.
% % 
% % % in cup_2 - need to detect the fingers, and at least one side of the cup
% % 
% % % in cup_3 - find where the cup meets the bottom lip. maybe the shape of
% % % the lips can help? Also, the small "corner" where the lips meet the mouth
% % % can be a distinctive shape.
% % 
% % % in straw - already worked on this, revert to previous things.
% % 
% % % bottle - similar to cup_3. 
% % 
% % % todo - apply the e.g, cup detector only in the lower part of the image,
% % % or at a place *attached* to skin, also with multiple rotation.
% % % align using flip / rigid /affine transformation between the example
% % % before learning the models.
% % 
% % curClusters = [curClusters_cup1,curClusters_cup3,curClusters_straw,curClusters_bottle];
% % qq = applyToSet(conf_new,curClusters,lipImages_test,[],'drinking_sanity',...
% %     'override',true,'disp_model',true,'rotations',-20:10:20);
% % 
% % [prec,rec,aps] = calc_aps(qq,t_test);
% % 
% % figure,plot(rec,prec)
% % 
% % 
% % 
% % newDets = combineDetections(qq);
% % [prec,rec,aps] = calc_aps(newDets,t_test)
% % 
% % plot(rec,prec)
% % 
% % 
% % 
% % %%
% % conf_new = conf;
% % conf_new.features.winsize = [6 6];
% % conf_new.detection.params.init_params.sbin = 8;
% % conf_new.features.vlfeat.cellsize = 8;
% % conf_new.detection.params.detect_levels_per_octave = 8;
% % conf_new.detection.params.init_params.sbin = conf_new.features.vlfeat.cellsize;
% % X = imageSetFeatures2(conf_new,lipImages_train(t_train),true,conf_new.features.winsize*8);
% % 
% % [C,IC] = vl_kmeans(X,20, 'NumRepetitions', 100);
% % d_clusters3 = makeClusterImages(lipImages_train(t_train),C,IC,X,...
% %     'drinking3_appearance_clusters');
% % 
% % % d_clusters3 = d_clusters3(1:8);
% % curSuffix = 'd_clusters3';
% % d_clusters3_trained= train_patch_classifier(conf_new,d_clusters3,...
% %     lipImages_train_2(~t_train),'suffix',curSuffix,'toSave',true,'override',true,'keepSV',true);
% % 
% % [prec,rec,aps] = calc_aps(qq,t_test)
% % plot(rec,prec)
% % plot(sort(aps),'.')
% % 
% % %%
% % 
% %  [A,AA] = visualizeClusters(conf_new,lipImages_test_2,qq(6),'add_border',...
% %       false,'nDetsPerCluster',...
% %         50,...
% %         'disp_model',true,'height',64);
% %     
% %     
% % 

