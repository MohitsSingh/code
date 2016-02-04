initpath;
config;
conf.class_subset = conf.class_enum.LOOKING_THROUGH_A_TELESCOPE;
initLandMarkData;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
for k = 1:length(train_ids)
    if (train_labels(k))
        conf.not_crop = true;
        imshow(getImage(conf,train_ids{k}));
        pause
    end
    
end



figure,imshow(mImage(train_faces(t_train)));

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


% % % [detector,q_train,q_test_cup3,conf3] = makeSpecializedDetector(conf,cup3SubImages,'cup3',override,...
% % %     cupInds_3_abs,bundle);
% % %     
% % % 
% % % q_train = arrangeDet(q_train,'index');
% % % q_test = arrangeDet(q_test_cup3,'index');
% % % 
% % % lipRects_test_on_faces2 = 32+lipBoxes_test_r_2/2;
% % % b = q_test.cluster_locs(:,1:4);
% % % [numRows numColumns area] = BoxSize(lipRects_test_on_faces2);
% % % res_test_rects = bsxfun(@times,b,numRows/100)+...
% % %     [lipRects_test_on_faces2(:,1:2) lipRects_test_on_faces2(:,1:2)];
% % % 
% % % lipRects_train_on_faces2 = 32+lipBoxes_train_r_2/2;
% % % b = q_train.cluster_locs(:,1:4);
% % % [numRows numColumns area] = BoxSize(lipRects_train_on_faces2);
% % % res_train_rects = bsxfun(@times,b,numRows/100)+...
% % %     [lipRects_train_on_faces2(:,1:2) lipRects_train_on_faces2(:,1:2)];
% % % 
% % %  feats_train = getBOWFeatures(conf,model,get_full_image,round(res_train_rects),descs_train); 
% % %  feats_test = getBOWFeatures(conf,model,test_faces_2,round(res_test_rects),descs_test);
% % %  
% % % rects1_lips = displayRectsOnImages(lipRects_test_on_faces2(t_test,:),test_faces_2(t_test));
% % % rects_dets = displayRectsOnImages(res_test_rects(t_test,:),test_faces_2(t_test));
% % % 
% % % % rects_dets = displayRectsOnImages(b(t_test,:),lipImages_test_100(t_test));
% % % 
% % % close all;
% % % mImage(test_faces_2(t_test));
% % % hold on;
% % % plotBoxes2(rects1_lips(:,[2 1 4 3]),'LineWidth',2,'Color','r');
% % % plotBoxes2(rects_dets(:,[2 1 4 3]),'LineWidth',2,'Color','g');
% % % %%
% % % 
% % % % for k = 1:length(get_full_image)
% % % %     
% % % % end
% % % 
% % % q_test_cup3 = arrangeDet(q_test_cup3,'index');
% % % % m_train = multiCrop(conf3,lipImages_train_100,round(q_train.cluster_locs),[50 50]);
% % % % m_test= multiCrop(conf3,lipImages_test_100,round(q_test.cluster_locs),[50 50]);
% % % 
% % % %[feats_train_straw,feats_test_straw] = extractBowFeats(conf,m_train,m_test,'lipFeats_straw.mat');
% % % 
% % % % x_train1 = imageSetFeatures2(conf3,m_train,true,[50 50]);
% % % % x_test1 = imageSetFeatures2(conf3,m_test,true,[50 50]);
% % % % 
% % % q_test = arrangeDet(q_test_cup3,'index');
% % % q_scores = q_test.cluster_locs(:,12);
% % % % 
% % % % [~,~,~,~,svm_model_2] = train_classifier(x_train1(:,strawInds_abs),x_train1(:,~t_train),[],[],2);
% % % % [~,~,svm_scores_2] = svmpredict(zeros(size(x_test1,2),1),double(x_test1'),svm_model_2);
% % % f_not = find(~t_train);
% % % f_not = f_not(1:5:end);
% % % [w,~,~,~,svm_model_3] = train_classifier(feats_train(:,cupInds_3_abs),feats_train(:,f_not),[],[],0);
% % % [~,~,svm_scores_3] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model_3);
% % % % svm_scores_3 = w'*feats_test;
% % % % [r,ir] = sort(svm_scores_1,'descend');
% % % % mImage(m_test(ir(1:50)));
% % % % svm_scores3 = normalise(svm_scores_3);
% % % % scores_t = q_scores+svm_scores1;
% % % %%
% % % %t_score_straw = +0*svm_scores1+1*q_scores(:)+1*svm_scores2+6*svm_scores3;%10*svm_scores1(:);
% % % % t_score_cup3 = double(q_test_cup3.cluster_locs(:,12))+0.5*double(svm_scores_3>0.5)';
% % % t_score_cup3 = double(q_test_cup3.cluster_locs(:,12))+1.5*double(svm_scores_3);
% % % %10*double(kp_score(:)>3);
% % % [r,ir] = sort(t_score_cup3,'descend');
% % % clf;
% % % %  rr = multiImage(lipImages_test_100(ir(1:50)));
% % % %  imshow(imresize(rr,.5));
% % % [prec,rec,aps] = calc_aps2(t_score_cup3,t_test);%,sum(test_labels))
% % % plot(rec,prec)
% % % % add some more features.
% % % 
% % % %%
% % % t_score = h_score(:)+.05*kp_score(:);
% % % %10*double(kp_score(:)>3);
% % % [r,ir] = sort(t_score,'descend');
% % %  mImage(lipImages_test(ir(1:50)));
% % % %%
% % % % [r,ir] = sort(kp_score,'descend');
% % % % mImage(lipImages_test(ir(1:50)));
% % % 
% % % 
% % % % [detector,q_train,q_test,conf3] = makeSpecializedDetector(conf,strawSubImages,'straw',lipImages_train_100,lipImages_test_100,...
% % % %     t_train(non_action_train),t_test,true);
% % % 
% % % train_res = visualizeLocs2_new(conf3,lipImages_train_100(t_train),q_train.cluster_locs);
% % % mImage(train_res);

override = false;
model.numSpatialX = [2];
model.numSpatialY = [2];
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

% add another svm score for the entire face region.

model2 = model;
model2.numSpatialX = [2 4];
model2.numSpatialY = [2 4];

feats_train = getBOWFeatures(conf,model2,get_full_image,[],descs_train);
feats_test = getBOWFeatures(conf,model2,test_faces_2,[],descs_test);


[~,~,~,~,svm_model] = train_classifier(feats_train(:,t_train),feats_train(:,~t_train));
[~,~,svm_scores] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model);


%%
d = 2;
cup1_scores = q_test_cup1.cluster_locs(:,12) + d*svm_scores_cup1;
cup2_scores = q_test_cup2.cluster_locs(:,12) + d*svm_scores_cup2;
cup3_scores = q_test_cup3.cluster_locs(:,12) + d*svm_scores_cup3;
bottle_scores = q_test_bottle.cluster_locs(:,12) + d*svm_scores_bottle;
straw_scores = q_test_straw.cluster_locs(:,12) + d*svm_scores_straw;

allScores = [cup1_scores,cup2_scores,cup3_scores,bottle_scores,straw_scores];
m = max(allScores,[],2)+0.5*svm_scores+10*double(test_face_scores(:)>-.81);
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

% % q_train_1 = arrangeDet(q_train_1,'index');
% mmm = visualizeLocs2_new(conf3,lipImages_train_100(~non_action_train),q_train_1.cluster_locs);
% t_train_2 = t_train(~non_action_train);
% t_train_2 = t_train_2(q_train_1.cluster_locs(:,11));
% % mImage(mmm(1:15:end));
% % mImage(mmm(t_train_2));
% 
% mmm_true = mmm(t_train_2);
% mmm_false = mmm(~t_train_2);
% mmm_true = mmm_true(1:9);
% 
% x_true = imageSetFeatures2(conf3,mmm_true,true,[]);
% x_false = imageSetFeatures2(conf3,mmm_false,true,[]);
% 
% C = 1;
% w1 = [];
% s = 2;
% [~,~,~,~,svm_model] = train_classifier(x_true,x_false,C,w1,s);
% 
% q_test = arrangeDet(q_test,'index');
% 
% mmm_test = visualizeLocs2_new(conf3,lipImages_test_100,q_test.cluster_locs);
% x_1_test = imageSetFeatures2(conf3,mmm_test,true,[]);
% [~,~,x_1_test_scores] = svmpredict(zeros(size(x_1_test,2),1),x_1_test',svm_model);
% % x_1_test_scores = d2'*x_1_test;
% % q_train = applyToSet(conf,detector,train_imgs(t_train),[],...
% %     suffix,'useLocation',0,'disp_model',true,'override',override);
% % test_faces_2 = test_faces(test_face_scores>=min_test_score);

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

