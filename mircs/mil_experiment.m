initpath;
config;

conf.class_subset = conf.class_enum.DRINKING;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

conf.max_image_size = inf;
% prepare the data...

%%

%%

load train_landmarks_full_face.mat; 
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat
load newFaceData4.mat

% remove the images where no faces were detected.
train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));
all_test_labels=all_test_labels(test_dets.cluster_locs(:,11));
all_train_labels=all_train_labels(train_dets.cluster_locs(:,11));

% and do the same for the ids.
% 
[faceLandmarks_train,lipBoxes_train,faceBoxes_train] = landmarks2struct(train_landmarks_full_face);
train_face_scores = [faceLandmarks_train.s];
[r_train,ir_train] = sort(train_face_scores,'descend');
[faceLandmarks_test,lipBoxes_test,faceBoxes_test] = landmarks2struct(test_landmarks_full_face);
test_face_scores = [faceLandmarks_test.s];
[r_test,ir_test] = sort(test_face_scores ,'descend');

train_labels_sorted = train_labels(ir_train);
test_labels_sorted = test_labels(ir_test);
m_train = multiImage(train_faces(ir_train(train_labels_sorted)),true);
% figure,imshow(m_train);
m_test = multiImage(test_faces(ir_test(test_labels_sorted)),true);
% figure,imshow(m_test);
% find 'true' lip coordinates.
debug_ = 1;

%%

%%


% get a random subset of the train faces and use logistic regression to
% score the faces..
imshow(multiImage(train_faces(ir_train(1:20:end))));
sss = r_train(1:20:end);
ttt = false(size(sss));
ttt([1:56 58:63 66:67 69 72 74 76 80 96:98 113 127 131 147:148 157 164 172 177]) = true;

[w_,b_] = logReg(sss(:),ttt(:));

% curScores = sigmoid(sss*w+ b);    
% plot(curScores);
% hold on;
% plot(ttt,'r+');
% 

% figure,plot(sigmoid(r_train*w_+b_))

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);

% missingTestInds = (test_face_scores < min_train_score) & test_labels';
% test_ids_d_1 = test_ids(test_dets.cluster_locs(:,11));
% missingTestRects = selectSamples(conf,test_ids_d_1(missingTestInds),'missingTestTrueFaces');
% missingTestRects = cat(1,missingTestRects{:});
% missingTestRects(:,4) = missingTestRects(:,4)+missingTestRects(:,2);
% missingTestRects(:,3) = missingTestRects(:,3)+missingTestRects(:,1);
% save missingTestRects.mat missingTestRects missingTestInds

%% 
close all;
train_faces = train_faces(train_face_scores>=min_train_score);
get_full_image = get_full_image(train_face_scores>=min_train_score);
train_faces_4 = train_faces_4(train_face_scores>=min_train_score);


test_faces = test_faces(test_face_scores>=min_test_score);
test_faces_2 = test_faces_2(test_face_scores>=min_test_score);
test_faces_4 = test_faces_4(test_face_scores>=min_test_score);


lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);

lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,[80 80],'both','abs'));

lipImages_train = multiCrop(conf,train_faces,lipBoxes_train_r_2,[50 50]);

lipImages_train_q = multiCrop(conf,get_full_image,32+round(lipBoxes_train_r_2/2),[50 50]);
lipImages_train2 = multiCrop(conf,train_faces_4,64+lipBoxes_train_r_2);
lipImages_test2 = multiCrop(conf,test_faces_4,64+lipBoxes_test_r_2);

save lipImagesHires.mat lipImages_train2 lipImages_test2

displayRectsOnImages(64+lipBoxes_train_r_2(1:50,:),train_faces_4(1:50));

% mImage(lipImages_train(1:50));
% mImage(lipImages_train_q(1:50));
% mImage(lipImages_train_q2);

% lipBoxes_train_r_3 = round(inflatebbox(lipBoxes_train_r,[100 100],'both','abs'));
% lipImages_train_2 = multiCrop(conf,train_faces,lipBoxes_train_r_3,[60 60]);


% figure,imshow(multiImage(lipImages_train(t_train),false));

%lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[64 64],'both','abs'));
lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[80 80],'both','abs'));

lipImages_test = multiCrop(conf,test_faces,lipBoxes_test_r_2,[50 50]);
% 
% lipBoxes_test_r_3 = round(inflatebbox(lipBoxes_test_r,[100 100],'both','abs'));
% lipImages_test_2 = multiCrop(conf,test_faces,lipBoxes_test_r_3,[60 60]);

%  figure,imshow(multiImage(lipImages_test_2(t_test),false));
% figure,imshow(multiImage(lipImages_test(t_test),false));
% 
% figure,imshow(multiImage(lipImages_test(~t_test),false));
% 
% % show train faces on 

% save lipData.mat lipImages_train lipImages_test t_train t_test train_faces get_full_image...
%     test_faces test_faces_2 t_train t_test;

%%
conf.suffix = 'rgb';
dict = learnBowDictionary(conf,train_faces,true);
model.numSpatialX = [2];
model.numSpatialY = [2];
model.kdtree = vl_kdtreebuild(dict) ;
model.quantizer = 'kdtree';
model.vocab = dict;
model.w = [] ;
model.b = [] ;
model.phowOpts = {'Color','RGB'};
% figure,imshow(getImage(conf,train_ids{train_dets.cluster_locs(550,11)}));

train_ids_d = train_ids(train_dets.cluster_locs(:,11));
train_ids_d = train_ids_d(train_face_scores>=min_train_score);
test_ids_d = test_ids(test_dets.cluster_locs(:,11));
test_ids_d = test_ids_d(test_face_scores>=min_test_score);

psix_train_local = getBOWFeatures(conf,model,lipImages_train);
psix_test_local = getBOWFeatures(conf,model,lipImages_test);
psix_train_face = getBOWFeatures(conf,model,train_faces);
psix_test_face = getBOWFeatures(conf,model,test_faces);

psix_train_face2 = getBOWFeatures(conf,model,get_full_image);
psix_test_face2 = getBOWFeatures(conf,model,test_faces_2);

conf.features.vlfeat.cellsize = 8;
flat = true;
hog_train_face = imageSetFeatures2(conf,train_faces,flat,[40 40]);
hog_test_face = imageSetFeatures2(conf,test_faces,flat,[40 40]);
hog_train_face_2 = imageSetFeatures2(conf,get_full_image,flat,1.5*[40 40]);
hog_test_face_2 = imageSetFeatures2(conf,test_faces_2,flat,1.5*[40 40]);
[hog_train_local,sz] = imageSetFeatures2(conf,lipImages_train,flat,[48 48]);
hog_test_local = imageSetFeatures2(conf,lipImages_test,flat,[48 48]);
hog_train_global_d = imageSetFeatures2(conf,train_ids_d,1,[64 48]);
hog_test_global_d = imageSetFeatures2(conf,test_ids_d,1,[64 48]);
                            
% 
%% add some saliency-driven features.
% mkdir('tmp_train');
% mkdir('tmp_test');
% multiWrite(train_faces,'tmp_train2');
% multiWrite(test_faces,'tmp_test2');
% % 
[train_sal] = multiRead('tmp_train_res','.png');
[test_sal] = multiRead('tmp_test_res','.png');
[test_sal2] = multiRead('tmp_test_res2','.png');

conf.features.vlfeat.cellsize = 8;
flat = true;
hog_train_face_sal = imageSetFeatures2(conf,train_sal,flat,[40 40]);
hog_test_face_sal = imageSetFeatures2(conf,test_sal,flat,[40 40]);

psix_train_face_sal = getBOWFeatures(conf,model,train_faces,train_sal);
psix_test_face_sal = getBOWFeatures(conf,model,test_faces,test_sal);


[psix_train_face2_sal] = getSalFeatures(conf,model,get_full_image,'tmp_train2','tmp_train_res2');
[psix_test_face2_sal] = getSalFeatures(conf,model,test_faces_2,'tmp_test2','tmp_test_res2');
 
train_sal_small = multiCrop(conf,train_sal,[],[16 16]);
test_sal_small = multiCrop(conf,test_sal,[],[16 16]);

lipSal_train = multiCrop(conf,train_sal,lipBoxes_train_r_2,[1 1]);
lipSal_test = multiCrop(conf,test_sal,lipBoxes_test_r_2,[1 1]);

for k = 1:length(train_sal_small)
    train_sal_small{k} = train_sal_small{k}(:);
    lipSal_train{k} = lipSal_train{k}(:);
    
end
train_sal_small = im2double(cat(2,train_sal_small{:}));
lipSal_train = im2double(cat(2,lipSal_train{:}));

for k = 1:length(test_sal_small)
    test_sal_small{k} = test_sal_small{k}(:);
    lipSal_test{k} = lipSal_test{k}(:);
    
end
test_sal_small = im2double(cat(2,test_sal_small{:}));
lipSal_test = im2double(cat(2,lipSal_test{:}));

% 
% [train_sal_global] = multiRead('~/data/jpeg_images_crop_res/','.png',train_ids_d);
% [test_sal_global] = multiRead('~/data/jpeg_images_crop_res/','.png',test_ids_d);
% 
% psix_train_sal_global = getBOWFeatures(conf,model,train_ids_d,train_sal_global);
% psix_test_sal_global = getBOWFeatures(conf,model,test_ids_d,test_sal_global);
% 

% figure,imshow(multiImage(train_sal_small(t_train),false))

save allFeats.mat   psix_train_local psix_test_local...
                                psix_train_face psix_test_face...
                                psix_train_face2 psix_test_face2...
                                hog_train_face hog_test_face...
                                hog_train_local hog_test_local...
                                hog_train_global_d hog_test_global_d...
                                hog_train_face_2 hog_test_face_2...
                                hog_train_face_sal hog_test_face_sal...
                                psix_train_face_sal psix_test_face_sal...
                                psix_train_face2_sal psix_test_face2_sal...
                                lipSal_train lipSal_test...
                                train_sal_small test_sal_small;
                                

% figure,imshow(multiImage(train_sal(t_train)))
% figure,imshow(multiImage(train_faces(t_train)))
%% and another channel which is bow x saliency.


%%
% choose a category
conf.class_subset = conf.class_enum.DRINKING;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));

t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);

% imshow(multiImage(test_faces(t_test)));

y_train = 2*(t_train==1)-1;
y_test = 2*(t_test==1)-1;

ss = '-t 0 -c .001 w1 1';
svmModel_bow_face= svmtrain(y_train, double(psix_train_face'),ss);
svmModel_bow_face2= svmtrain(y_train, double(psix_train_face2'),ss);

svmModel_bow_face_sal= svmtrain(y_train, double(psix_train_face_sal'),ss);

svmModel_bow_face2_sal= svmtrain(y_train, double(psix_train_face2_sal'),ss);

% svmModel_hog_ext = svmtrain(y_train, double(hog_train_ext'),ss);

svmModel_bow_local= svmtrain(y_train, double(psix_train_local'),ss);
svmModel_hog_face= svmtrain(y_train, double(hog_train_face'),ss);
svmModel_hog_face_sal= svmtrain(y_train, double(hog_train_face_sal'),ss);

svmModel_hog_face_2= svmtrain(y_train, double(hog_train_face_2'),ss);
svmModel_hog_local= svmtrain(y_train, double(hog_train_local'),ss);
svmModel_hog_global= svmtrain(y_train, double(hog_train_global_d'),ss);

svmModel_sal_patch= svmtrain(y_train, double(train_sal_small'),ss);
svmModel_lipSal_patch= svmtrain(y_train, double(lipSal_train'),ss);

% svmModel_global_sal= svmtrain(y_train, double(psix_train_sal_global'),ss);


% add some more "specialized" detectors...
% figure,imshow(multiImage(train_faces(t_train)))

% bow scan does bow scanning window.
f = find(t_train);
f_test = find(t_test);
%%
% img = test_faces{f_test(7)};
% figure,subplot(2,1,1);imshow(img);
%  [ hists, q ] = bowScan( conf,model, img);
%  
%  [~, ~, scan_1] = svmpredict(zeros(9,1),double(hists'),svmModel_bow_local);
% qq = zeros(size(q));
%  for z = 1:length(unique(q))
%      qq(q==z)=scan_1(z);
%  end
% 
%  subplot(2,1,2),imagesc(qq);

allTrainScores = zeros(9,length(train_faces));
allTestScores = zeros(9,length(train_faces));
z = zeros(9,1);
 parfor k = 1:length(train_faces)
     k
      [ hists] = bowScan( conf,model, train_faces{k});
       [~, ~, allTrainScores(:,k)] = svmpredict(z,double(hists'),svmModel_bow_local);
 end
 
 parfor k = 1:length(test_faces)train
     k
      [ hists] = bowScan( conf,model, test_faces{k});
       [~, ~, allTestScores(:,k)] = svmpredict(z,double(hists'),svmModel_bow_local);
 end
    

 %%

x = hog_train_face(:,t_train);
[C,IC] = vl_kmeans(x,10, 'NumRepetitions', 100);
d_clusters = makeClusterImages(train_faces(t_train),C,IC,x,...
    'hogTrainFaceClusters');

% rects_drink_obj = selectSamples(conf,train_faces(t_train),'drinkObj');
% save rects_drink_obj rects_drink_obj;
load rects_drink_obj;



cc = cat(2,c.cluster_samples);
a_c = visualizeLocs2_new(conf,train_faces,cat(1,c.cluster_locs));
[C,IC] = vl_kmeans(cc,5, 'NumRepetitions', 100);
d_clusters = makeClusterImages(a_c,C,IC,cc,...
    'drinking_appearance_clusters');

d_clusters_trained = train_patch_classifier(conf,d_clusters,train_faces(~t_train),'suffix','drinkers','override',false);

for k = 1:length(d_clusters_trained)
    imshow(showHOG(conf,d_clusters_trained(k)));
end

d_clusters_trained_res = applyToSet(conf,d_clusters_trained,train_faces,[],'drinkers_res','override',false,...
    'rotations',-20:10:20);

d_clusters_test_res = applyToSet(conf,d_clusters_trained,test_faces,[],'drinkers_res2',...
    'override',false,'rotations',-20:10:20);

[prec,rec,aps] = calc_aps(d_clusters_test_res,t_test,sum(test_labels));
plot(rec,prec)



%% learn cup/bottle appearance...
figure,imshow(multiImage(get_full_image(t_train)));

tt = get_full_image(t_train);
tt([41 48]) = [];
rects_drink_obj2 = selectSamples(conf,tt,'drinkObj2');
save rects_drink_obj2 rects_drink_obj2;

conf.features.winsize = [5 5];
c = rects2clusters(conf,rects_drink_obj2,tt,[],0);

cc = cat(2,c.cluster_samples);


a_c = visualizeLocs2_new(conf,tt,cat(1,c.cluster_locs));
[C,IC] = vl_kmeans(cc,5, 'NumRepetitions', 100);
d_clusters2 = makeClusterImages(a_c,C,IC,cc,...
    'drinking2_appearance_clusters');

a = cat(1,c.cluster_locs);
Zs = createConsistencyMaps(c,[128 128],1:62,inf,[19 5]);

Z = zeros(128);
for k = 1:length(Zs)
    Z = Z+Zs{k};
end


%d_clusters2_trained = train_patch_classifier(conf,d_clusters2,get_full_image(~t_train),'suffix','drinkers2','override',true);
% train using the "hard" negatives classes....

hardSet = ismember(t_train_all, [conf.class_enum.BRUSHING_TEETH,conf.class_enum.BLOWING_BUBBLES,...
    conf.class_enum.SMOKING]);

conf.clustering.num_hard_mining_iters = 10;
d_clusters2_trained = train_patch_classifier(conf,d_clusters2,get_full_image(hardSet),'suffix','drinkers2','override',true);

for k = 1:length(d_clusters2_trained)
    imshow(showHOG(conf,d_clusters2_trained(k)));pause;
end

ZZ = repmat({Z},5,1);

d_clusters2_trained_res = applyToSet(conf,d_clusters2_trained,get_full_image,[],'drinkers2_res','override',true,...
    'rotations',0,'useLocation',ZZ);
    %'rotations',-20:10:20);

d_clusters2_test_res = applyToSet(conf,d_clusters2_trained,test_faces_2,[],'drinkers2_res2',...
    'override',true,'rotations',0,'useLocation',ZZ);

[prec,rec,aps] = calc_aps(d_clusters2_test_res,t_test,sum(test_labels));
plot(rec,prec)

%%
A = visualizeLocs2_new(conf,test_faces,d_clusters_test_res(2).cluster_locs);

% figure,imshow(multiImage(A(1:50)))

% dets = combineDetections(d_clusters_test_res);

[prec2,rec2,aps2] = calc_aps(dets,t_test,sum(test_labels));


boxes = cat(1,c.cluster_locs);
z = ones(128);
figure,imagesc(z);axis image;
real_centers = boxCenters(boxes);
lip_centers = boxCenters(lipBoxes_train_t(t_train,:));

hold on;
plot(real_centers(:,1)-lip_centers(:,1),real_centers(:,2)-lip_centers(:,2),'r+')

% aa = cat(1,c.cluster_locs);
% a_c = visualizeLocs2_new(conf,train_faces(t_train),aa);
figure,imshow(multiImage(lipImages_train,false))% score the faces..
imshow(multiImage(train_faces(ir_train(1:20:end))));
sss = r_train(1:20:end);
ttt = false(size(sss));
ttt([1:56 58:63 66:67 69 72 74 76 80 96:98 113 127 131 147:148 157 164 172 177]) = true;

[w_,b_] = logReg(sss(:),ttt(:));

% curScores = sigmoid(sss*w+ b);    
% plot(curScores);
% hold on;
% plot(ttt,'r+');
% 

figure,plot(sigmoid(r_train*w_+b_))

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);

% missingTestInds = (test_face_scores < min_train_score) & test_labels';
% test_ids_d_1 = test_ids(test_dets.cluster_locs(:,11));
% missingTestRects = selectSamples(conf,test_ids_d_1(missingTestInds),'missingTestTrueFaces');
% missingTestRects = cat(1,missingTestRects{:});
% missingTestRects(:,4) = missingTestRects(:,4)+missingTestRects(:,2);
% missingTestRects(:,3) = missingTestRects(:,3)+missingTestRects(:,1);
% save missingTestRects.mat missingTestRects missingTestInds


figure,imshow(multiImage(lipImages_test(t_test),false));

theClusts = makeClusters(cc,[]);
conf.features.winsize = [8 8];
%clusters_t = train_patch_classifier(conf,c,train_faces(~t_train),'suffix','drinkers2','override',true);
clusters_t = train_patch_classifier(conf,theClusts,train_faces(~t_train),'suffix','drinkers2','override',true);
conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_min_scale = .8;
[q_tt] = applyToSet(conf,clusters_t,train_faces,[],'drinkers_train2','override',true);
[q_t] = applyToSet(conf,clusters_t,test_faces,[],'drinkers_test2','override',true);

tt_faces = train_faces(t_train);
tt_faces_f = train_faces(~t_train);

discovery_sets = { tt_faces(1:2:end),tt_faces(2:2:end)};
natural_sets = { tt_faces_f(1:2:end),tt_faces_f(2:2:end)};
clusters=refineClusters(conf,clusters_t,discovery_sets,natural_sets,'drinkers_refine');

[q_tt2] = applyToSet(conf,clusters,train_faces,[],'drinkers_train2_ref','override',false);
[q_t2] = applyToSet(conf,clusters,test_faces,[],'drinkers_test2_ref','override',false);

[~,~,qq_train] = combineDetections(q_tt);
[~,~,qq_test] = combineDetections(q_t);

ws = [];
for q = 1:size(C,2)
    q
    [w] = train_classifier(d_clusters(q).cluster_samples ,hog_train_face(:,~t_train),.01,10);
    ws = [ws,w];
end
% 
%%
% ws = [];
[~, ~, dt_train_1] = svmpredict(zeros(size(t_train,1),1),double(psix_train_face'),svmModel_bow_face);
dt_train_1 = dt_train_1*svmModel_bow_face.Label(1);
[~, ~, dt_train_2] = svmpredict(zeros(size(t_train,1),1),double(psix_train_local'),svmModel_bow_local);
dt_train_2 = dt_train_2*svmModel_bow_local.Label(1);
[~, ~, dt_train_3] = svmpredict(zeros(size(t_train,1),1),double(hog_train_face'),svmModel_hog_face);
dt_train_3 = dt_train_3*svmModel_hog_face.Label(1);
[~, ~, dt_train_4] = svmpredict(zeros(size(t_train,1),1),double(hog_train_local'),svmModel_hog_local);
dt_train_4 = dt_train_4*svmModel_hog_local.Label(1);
[~, ~, dt_train_5] = svmpredict(zeros(size(t_train,1),1),double(hog_train_global_d'),svmModel_hog_global);
dt_train_5 = dt_train_5*svmModel_hog_global.Label(1);
[~, ~, dt_train_6] = svmpredict(zeros(size(t_train,1),1),double(hog_train_face_2'),svmModel_hog_face_2);
dt_train_6 = dt_train_6*svmModel_hog_face_2.Label(1);
[~, ~, dt_train_7] = svmpredict(zeros(size(t_train,1),1),double(hog_train_face_sal'),svmModel_hog_face_sal);
dt_train_7 = dt_train_7*svmModel_hog_face_sal.Label(1);
[~, ~, dt_train_8] = svmpredict(zeros(size(t_train,1),1),double(psix_train_face_sal'),svmModel_bow_face_sal);
dt_train_8 = dt_train_8*svmModel_bow_face_sal.Label(1);
[~, ~, dt_train_9] = svmpredict(zeros(size(t_train,1),1),double(train_sal_small'),svmModel_sal_patch);
dt_train_9 = dt_train_9*svmModel_sal_patch.Label(1);
dt_train_10 = double(lipSal_train');
dt_train_11 = sigmoid(train_face_scores(train_face_scores>=min_train_score)*w_+b_);
% dt_train_12 = min(allTrainScores)';
% [~, ~, dt_train_13] = svmpredict(zeros(size(t_train,1),1),double(psix_train_sal_global'),svmModel_global_sal);
% [~, ~, dt_train_14] = svmpredict(zeros(size(t_train,1),1),double(hog_train_ext'),svmModel_hog_ext);

[~, ~, dt_train_12] = svmpredict(zeros(size(t_train,1),1),double(psix_train_face2'),svmModel_bow_face2);
dt_train_12 = dt_train_12*svmModel_bow_face2.Label(1);
[~, ~, dt_train_13] = svmpredict(zeros(size(t_train,1),1),double(psix_train_face2_sal'),svmModel_bow_face2_sal);
dt_train_13 = dt_train_13*svmModel_bow_face2_sal.Label(1);

% [~,~,dt_train_14]  = combineDetections(d_clusters_trained_res);
% [~,~,dt_train_15]  = combineDetections(d_clusters2_trained_res);

% dt_train_999 = (ws'*hog_train_face)';

[~, ~, dt_test_1] = svmpredict(zeros(size(t_test,1),1),double(psix_test_face'),svmModel_bow_face);
dt_test_1 = dt_test_1*svmModel_bow_face.Label(1);
[~, ~, dt_test_2] = svmpredict(zeros(size(t_test,1),1),double(psix_test_local'),svmModel_bow_local);
dt_test_2 = dt_test_2*svmModel_bow_local.Label(1);
[~, ~, dt_test_3] = svmpredict(zeros(size(t_test,1),1),double(hog_test_face'),svmModel_hog_face);
dt_test_3 = dt_test_3*svmModel_hog_face.Label(1);
[~, ~, dt_test_4] = svmpredict(zeros(size(t_test,1),1),double(hog_test_local'),svmModel_hog_local);
dt_test_4 = dt_test_4*svmModel_hog_local.Label(1);
[~, ~, dt_test_5] = svmpredict(zeros(size(t_test,1),1),double(hog_test_global_d'),svmModel_hog_global);
dt_test_5 = dt_test_5*svmModel_hog_global.Label(1);
[~, ~, dt_test_6] = svmpredict(zeros(size(t_test,1),1),double(hog_test_face_2'),svmModel_hog_face_2);
dt_test_6 = dt_test_6*svmModel_hog_face_2.Label(1);
[~, ~, dt_test_7] = svmpredict(zeros(size(t_test,1),1),double(hog_test_face_sal'),svmModel_hog_face_sal);
dt_test_7 = dt_test_7*svmModel_hog_face_sal.Label(1);
[~, ~, dt_test_8] = svmpredict(zeros(size(t_test,1),1),double(psix_test_face_sal'),svmModel_bow_face_sal);
dt_test_8 = dt_test_8*svmModel_bow_face_sal.Label(1);
[~, ~, dt_test_9] = svmpredict(zeros(size(t_test,1),1),double(test_sal_small'),svmModel_sal_patch);
dt_test_9 = dt_test_9*svmModel_sal_patch.Label(1);
dt_test_10 = double(lipSal_test');
dt_test_11 = sigmoid(test_face_scores(test_face_scores>=min_test_score)*w_+b_);

% [~, ~, dt_test_14] = svmpredict(zeros(size(t_test,1),1),double(hog_test_ext'),svmModel_hog_ext);

[~, ~, dt_test_12] = svmpredict(zeros(size(t_test,1),1),double(psix_test_face2'),svmModel_bow_face2);
dt_test_12 = dt_test_12*svmModel_bow_face2.Label(1);
[~, ~, dt_test_13] = svmpredict(zeros(size(t_test,1),1),double(psix_test_face2_sal'),svmModel_bow_face2_sal);
dt_test_13 = dt_test_13*svmModel_bow_face2_sal.Label(1);

% [~,~,dt_test_14]  = combineDetections(d_clusters_test_res);
% [~,~,dt_test_15]  = combineDetections(d_clusters2_test_res);


% dt_test_12 = min(allTestScores)';
% [~, ~, dt_test_13] = svmpredict(zeros(size(t_test,1),1),double(psix_test_sal_global'),svmModel_global_sal);
% dt_test_999 = (ws'*hog_test_face)';

% HERE-------------------
%%
ddd = 1;

% dt_train_999 = [];
% dt_test_999 = [];
% 
feat_train_all = [dt_train_1(:),dt_train_2(:),dt_train_3(:),dt_train_4(:),dt_train_5(:),dt_train_6(:),dt_train_7,dt_train_8,dt_train_9,dt_train_10,dt_train_11(:),dt_train_12,dt_train_13,dt_train_14,dt_train_15,train_faces_scores_r(:)];
feat_train_all = double(feat_train_all);

%sel = [1 2 4 5 7 8 9 10 11 12 14:16];

sel = [1 2 4 5 7 8 9 10 12 15:18 21:23];

% sel = [1];  % bow face - 18%

% sel = [3]; % hog face - 18&
% sel = [12];

sel_add = [1:10];
% sel_add = [];
feat_train_all = [feat_train_all(:,sel),double(dt_train_999(:,sel_add))];

% [r,ir] = sort(dt_train_5,'ascend');
% figure,imshow(multiImage(train_faces(ir(1:50)),false))
% [r,ir] = sort(dt_test_5,'ascend');
% figure,imshow(multiImage(test_faces(ir(1:50)),false))

means  = mean(feat_train_all);
vars = std(feat_train_all);

feat_train_all = bsxfun(@minus,feat_train_all,means);
feat_train_all = bsxfun(@rdivide,feat_train_all,vars);
% feat-train-all-dont-care-removed
% fta_dcr = feat_train_all(~t_train_dontcare,:);
% y_train_dcr = y_train(~t_train_dontcare);
% 
% fta_dcr = feat_train_all(t_train_dontcare| t_train,:);
% y_train_dcr = y_train(t_train_dontcare | t_train);

% svmModel_all= svmtrain(y_train_dcr,fta_dcr,'-t 0 -c .00001 w1 1');

svmModel_all= svmtrain(y_train,feat_train_all,'-t 1 -c .01 w1 1');

[~, ~, decision_values_train] = svmpredict(zeros(size(t_train,1),1),double(feat_train_all),svmModel_all);
decision_values_train = decision_values_train.*svmModel_all.Label(1);
% [r,ir] = sort(decision_values_train,'ascend');
% figure,imshow(multiImage(get_full_image(ir(1:100)),false));
% 
% 
% svmModel_bow_face2_c = svmtrain(y_train(ir(1:100)), double(psix_train_face2(:,1:100)'),ss);
% 

% [~, ~, dt_test_12_c] = svmpredict(zeros(size(t_test,1),1),double(psix_test_face2'),svmModel_bow_face2_c);
% 
% dt_test_3_with_flip = max(dt_test_3,dt_test_3_flipped);
dt_test_3_with_flip = dt_test_3;
feat_test_all = [dt_test_1(:),ddd*dt_test_2(:),dt_test_3_with_flip(:),dt_test_4(:),dt_test_5(:),dt_test_6(:),dt_test_7(:),dt_test_8(:),dt_test_9(:),dt_test_10(:),dt_test_11(:),dt_test_12,dt_test_13,dt_test_14,dt_test_15,test_faces_scores_r(:)];
feat_test_all = [feat_test_all(:,sel),double(dt_test_999(:,sel_add))];
feat_test_all = double(feat_test_all);

feat_test_all = bsxfun(@minus,feat_test_all,means);
feat_test_all = bsxfun(@rdivide,feat_test_all,vars);

% svmModel_all= svmtrain(y_test, feat_test_all,'-t 0 -c 100 w1 1');

[~, ~, decision_values_test] = svmpredict(zeros(size(t_test,1),1),double(feat_test_all),svmModel_all);
decision_values_test = decision_values_test*svmModel_all.Label(1);

 %decision_values_test(f) =0;
% decision_values_test(f) = rand(size(f))*(max(decision_values_test)-min(decision_values_test))+min(decision_values_test);
scores_ = decision_values_test;%(D_M_test_*ws);

scores_bu = scores_;
tt = scores_bu>-.9;
scores_bu(tt) = scores_bu(tt)-dt_test_12(tt)*100;


% ttt_bool = false(size(tt));
% ttt_bool(ttt) = true;

ttt_scores = zeros(size(tt));
ttt_scores(ttt) = test_cigar_cup*.5;
ttt_scores(~tt) = mean(test_cigar_cup);

% scores_bu= scores_bu + ttt_scores;


[r,ir] = sort(scores_bu,'descend');
% plot(cumsum(rr(ir)))

% plot(cumsum(t_test(ir)))

% test_faces_p = paintRule(test_faces,t_test,[],[],3);
% figure,imshow(multiImage(test_faces(ir(1:100)),conf.classes(t_test_all(ir(1:100)))))

% imwrite(multiImage(test_faces_2(ir(1:300)),conf.classes(t_test_all(ir(1:300)))),'a.jpg')


imwrite(multiImage(test_faces(ir(1:100)),conf.classes(t_test_all(ir(1:100)))),'a.jpg')


%  imwrite(multiImage(test_sal2(ir(1:300)),conf.classes(t_test_all(ir(1:300)))),'b.jpg')

% t_ir = t_test(ir);
%  figure,imshow(multiImage(test_faces(ir((t_ir))),false))
% 
% imwrite(multiImage(test_faces_p(ir(1:300))),'drink_results.tif');

% mm = multiImage(test_faces(ir(1:150)));
% imwrite(mm,'docs/brushing_results_using_mouth.tif');
%  figure,imshow(multiImage(lipImages_test(ir(1:140)),false))

% 

%t_test2 = t_test_all == conf.class_enum.DRINKING;

posSet = [conf.class_enum.DRINKING];

test_labels_all = ismember(all_test_labels,posSet);

t_test2 = double(ismember(t_test_all, posSet));



% ignoreSet = ismember(t_test_all, [conf.class_enum.BRUSHING_TEETH,conf.class_enum.BLOWING_BUBBLES,...
%     conf.class_enum.SMOKING]);

% ignoreSet = ismember(t_test_all, [conf.class_enum.BRUSHING_TEETH,conf.class_enum.BLOWING_BUBBLES,...
%     conf.class_enum.SMOKING]);

ignoreSet = [];
t_test2(ignoreSet) = [];
scores_bu(ignoreSet) = [];

[prec,rec,aps] = calc_aps2(scores_bu,t_test2,sum(test_labels_all));

% [prec,rec,aps] = calc_aps2(scores_(~t_test_dontcare),t_test(~t_test_dontcare),sum(test_labels(~test_dontcare)));
figure,plot(rec,prec); title(num2str(aps));

% plot on each face the relative lip-box....

%%

%%

f = train_faces(t_train);
f_ = train_sal(t_train);

%%
for k = 1:length(f)
    
    %     labels = getMultipleSegmentations(f{1});
    sigma_=.8;
    minSize = 15;
    im = f{k};
    curIm = uint8(rgb2lab(im));
    sal_map = im2double(f_{k});
    labels = mexFelzenSegmentIndex(im, sigma_, minSize, 20);
    
    %figure,imagesc(labels)
    rr = regionprops(labels,sal_map,'PixelIdxList','Area','MeanIntensity');
    Z = paintRegionProps(labels,rr,[rr.MeanIntensity].*([rr.Area].^.5));
    clf;
    subplot(2,2,1);
    imagesc(im);
    subplot(2,2,2);
    imagesc(sal_map);
    subplot(2,2,3);
    imagesc(labels);
    subplot(2,2,4);
    imagesc(Z);
    pause;
end
    %%
% figure,imagesc(Z)


%ttt = train_ids_d_t(t_train);
% ttt = lipImages_train(t_train);
%%
ttt = get_full_image(t_train);
for k = 1:length(ttt)
    clf;
    im = im2uint8(getImage(conf,ttt{k}));    
%     im = imfilter(im,fspecial('gauss',9,2));
%     myGetSkinRegions(im);
    %bb = [1 1 size(im,2) size(im,1)];
%     bb = faceBoxes_train_1(:,k);
%      myGetSkinRegions(im,round(bb));
    skinprob = computeSkinProbability(double(im));
    r0  = im;
    r1 = im2uint8(jettify(skinprob));
    imshow([r0,r1]);
    pause;
end

% 
% imshow(train_faces{1})
% im = train_faces{1};
% myGetSkinRegions(im);

%% figure,imshow(multiImage(test_faces(ir(1:50))))
scores_ = -decision_values_test;%(D_M_test_*ws);
[r,ir] = sort(scores_,'descend');
% figure,imshow(multiImage(test_faces(ir(1:100))))
[prec,rec,aps] = calc_aps2(scores_,t_test,sum(test_labels))
figure,plot(rec,prec); title(num2str(aps));
%% figure,imshow(multiImage(test_faces(ir(1:100))))
figure,imshow(multiImage(test_faces(ir(1:50))))
% figure,imshow(multiImage(lipImages_test(ir(1:150)))

%%
ddd = 1;

feat_train_all = [dt_train_1(:),dt_train_2(:),dt_train_3(:),dt_train_4(:),dt_train_5(:)];

t = classregtree(feat_train_all,t_train(2:2:end));

% [r,ir] = sort(dt_train_5,'ascend');
% figure,imshow(multiImage(train_faces(ir(1:50)),false))
% [r,ir] = sort(dt_test_5,'ascend');
% figure,imshow(multiImage(test_faces(ir(1:50)),false))

means  = mean(feat_train_all);
vars = std(feat_train_all);

feat_train_all = bsxfun(@minus,feat_train_all,means);
feat_train_all = bsxfun(@rdivide,feat_train_all,vars);


svmModel_all= svmtrain(y_train(2:2:end),feat_train_all,'-t 0 -c .1 w1 1');

feat_test_all = [[dt_test_1(:),ddd*dt_test_2(:),dt_test_3(:),dt_test_4(:)],dt_test_5(:)];

feat_test_all = bsxfun(@minus,feat_test_all,means);
feat_test_all = bsxfun(@rdivide,feat_test_all,vars);

% svmModel_all= svmtrain(y_test, feat_test_all,'-t 0 -c 100 w1 1');

[predicted_label, ~, decision_values_test] = svmpredict(zeros(size(t_test,1),1),double(feat_test_all),svmModel_all);

%decision_values_test = decision_values_test;
decision_values_test = dt_test_1(:);



% svmModel_all= svmtrain(y_test, feat_train_all,'-t 0 w1 1');
% decision_values_test = feat_test_all*[1 1 .001*([10 1])]';
% figure,imshow(multiImage(test_faces(ir(1:50))))
scores_ = -decision_values_test;%(D_M_test_*ws);

% [yfit,nodes,cnums] = eval(t,feat_test_all);
% scores_ = cnums;
[r,ir] = sort(scores_,'descend');
% figure,imshow(multiImage(test_faces(ir(1:100))))
[prec,rec,aps] = calc_aps2(scores_,t_test,sum(test_labels))
figure,plot(rec,prec); title(num2str(aps));
%% figure,imshow(multiImage(test_faces(ir(1:50))))
scores_ = -decision_values_test;%(D_M_test_*ws);
[r,ir] = sort(scores_,'descend');
% figure,imshow(multiImage(test_faces(ir(1:100))))
[prec,rec,aps] = calc_aps2(scores_,t_test,sum(test_labels))
figure,plot(rec,prec); title(num2str(aps));
%% figure,imshow(multiImage(test_faces(ir(1:100))))
figure,imshow(multiImage(test_faces(ir(1:50))))
figure,imshow(multiImage(lipImages_test(ir(1:150))))
%%
% m = train_faces(t_train);
m = test_faces(ir);
%%
% m = lipImages_train(~t_train);
% imshow(m{1})
for k = 1:length(m)
    
% for k = 11
    im = m{k};
    im = im2double(im);
    im = imfilter(im,fspecial('gauss',5, 2));
%     figure,imshow(im);
im_1 = vl_xyz2luv(vl_rgb2xyz(im));
im_1 = (rgb2hsv(im));
% im_1 = im;
% lipColors = reshape(im2double(mm{k}),[],size(mm{k},3));
% imshow(im2double(mm{k}))
imcolors = reshape(im_1,[],size(im,3));
obj = gmdistribution.fit(imcolors,1);

h = pdf(obj,imcolors);
Z0 = reshape(h,size(im,1),size(im,2));

r0 = im; 
%r1 = im2double(jettify(Z0)); 
Z0 = Z0/max(Z0(:));
r1 = repmat(Z0,[1 1 3]);
r2 = edge(Z0,'canny');

r2 = repmat(r2,[1 1 3]);
r3 = edge(rgb2gray(im),'canny');
r3 = repmat(r3,[1 1 3]);
bsize = [3 3];
B = [];
for c= 1:3
    B = [B;im2col(im_1(:,:,c),bsize,'distinct')];
end

b_d = l2(B',B');

[r,ir] = sort(b_d,2,'ascend');
r = r(:,2:end);
%
knn = 5;
sigma_ = .1;
z = sum(exp(-r(:,1:knn)/sigma_),2)/knn;
z = repmat(z',prod(bsize),1);
sz = size(im);

b_z = col2im(z,bsize,sz(1:2),'distinct');
b_z= b_z-min(b_z(:));
b_z = b_z/max(b_z(:));
b_z = b_z.^2;
r4 = repmat(b_z,[1 1 3]);

% figure,imagesc(b_z);
r4 = jettify(b_z);
imshow([r0 r1 r2 r3 r4]);


pause;
end


% self similarity...

% imshow(im)
% p = im(65:69,37:41,:);
% imshow(p)


%%
mkdir('~/cropped');
parfor k = 1:length(train_ids)
    I = getImage(conf,train_ids{k});
    imwrite(I,fullfile('~/cropped',train_ids{k}));
end


parfor k = 1:length(test_ids)
    I = getImage(conf,test_ids{k});
    imwrite(I,fullfile('~/cropped',test_ids{k}));
end



%%
cigarSetTrain = find(ismember(t_train_all,conf.class_enum.SMOKING));
cupSetTrain = find(ismember(t_train_all,conf.class_enum.DRINKING));
brushSetTrain = find(ismember(t_train_all,conf.class_enum.BRUSHING_TEETH));
blowSetTrain = find(ismember(t_train_all,conf.class_enum.BLOWING_BUBBLES));
phoneSetTrain = find(ismember(t_train_all,conf.class_enum.PHONING));

cigarSetTest = find(ismember(t_test_all,conf.class_enum.SMOKING));
cupSetTest = find(ismember(t_test_all,conf.class_enum.DRINKING));
brushSetTest = find(ismember(t_test_all,conf.class_enum.BRUSHING_TEETH));
blowSetTest = find(ismember(t_test_all,conf.class_enum.BLOWING_BUBBLES));
phoneSetTest = find(ismember(t_test_all,conf.class_enum.PHONING));

imshow(multiImage(test_faces_2(phoneSetTest)))

cigarTrain_w = [1 2 3 4 7 10:13 17:19 21:25 29:34 36 37 40 41 43 44 46 50:54 56 59 61 62 64 66 68 70 71 72 74 75 77];
cupTrain_w = setdiff(1:length(cupSetTrain),[2 41 48]);
brushTrain_w = setdiff(1:length(brushSetTrain),[11 24 41 52 ]);
blowTrain_w = setdiff(1:length(blowSetTrain),[14 22 24 26 29 30 34 36 38 49 52 64 65 67]);
phoneTrain_w = setdiff(1:length(phoneSetTrain),[3 21]);


cigarTest_w = setdiff(1:length(cigarSetTest),[6 13 19 21 22 23 26 31 33 37 40 42 43 45 46 48 50 52 57 60 64 65 68 71 72 76 78 83 85 86 88 89 97]);
cupTest_w = setdiff(1:length(cupSetTest),[23 51 59 79 84 97]);
brushTest_w = setdiff(1:length(brushSetTest),76);
blowTest_w = setdiff(1:length(blowSetTest),[4 14 15 17 24 28 40 51 53 57 58 63 65 69 73 79 81 83 84 86 96 105 108]);
phoneTest_w = 1:length(phoneSetTest);

train_faces_cigar = get_full_image(cigarSetTrain(cigarTrain_w));
train_faces_cup = get_full_image(cupSetTrain(cupTrain_w));
train_faces_brush = get_full_image(brushSetTrain(brushTrain_w));
train_faces_blow = get_full_image(blowSetTrain(blowTrain_w));
train_faces_phone = get_full_image(phoneSetTrain(phoneTrain_w));

test_faces_cigar = test_faces_2(cigarSetTest(cigarTest_w));
test_faces_cup = test_faces_2(cupSetTest(cupTest_w));
test_faces_brush = test_faces_2(brushSetTest(brushTest_w));
test_faces_blow = test_faces_2(blowSetTest(blowTest_w));
test_faces_phone = test_faces_2(phoneSetTest(phoneTest_w));

train_cigar_rects = selectSamples(conf,train_faces_cigar,'train_cigar_rect_dir');
train_cups_rects = selectSamples(conf,train_faces_cup,'train_cup_rect_dir');
train_brush_rects = selectSamples(conf,train_faces_brush,'train_brush_rect_dir');
% train_blow_rects = selectSamples(conf,train_faces_blow,'train_blow_rect_dir');
train_phone_rects = selectSamples(conf,train_faces_phone,'train_phone_rect_dir');


test_cigar_rects = selectSamples(conf,test_faces_cigar,'test_cigar_rect_dir');
test_cups_rects = selectSamples(conf,test_faces_cup,'test_cup_rect_dir');
test_brush_rects = selectSamples(conf,test_faces_brush,'test_brush_rect_dir');
% test_blow_rects = selectSamples(conf,test_faces_blow,'test_blow_rect_dir');
test_phone_rects = selectSamples(conf,test_faces_phone,'test_phone_rect_dir');


train_cigar_rects_1 = imrect2rect(train_cigar_rects);
train_cup_rects_1 = imrect2rect(train_cups_rects);
train_brush_rects_1 = imrect2rect(train_brush_rects);
train_phone_rects_1 = imrect2rect(train_phone_rects);

test_cigar_rects_1 = imrect2rect(test_cigar_rects);
test_cup_rects_1 = imrect2rect(test_cups_rects);
test_brush_rects_1 = imrect2rect(test_brush_rects);
test_phone_rects_1 = imrect2rect(test_phone_rects);


ttt = [cupSetTest(cupTest_w);cigarSetTest(cigarTest_w)]

model2 = model;
model2.numSpatialX = [2 4];
model2.numSpatialY = [2 4];

sz = [60 60];
conf.features.vlfeat.cellsize = 8;
train_cigar_images = multiCrop(conf,train_faces_cigar,round(imrect2rect(train_cigar_rects)),sz);
train_cup_images = multiCrop(conf,train_faces_cup,round(imrect2rect(train_cups_rects)),sz);
train_brush_images = multiCrop(conf,train_faces_brush,round(imrect2rect(train_brush_rects)),sz);
%train_blow_images = multiCrop(conf,train_faces_blow,round(imrect2rect(train_blow_rects)),sz);
train_phone_images = multiCrop(conf,train_faces_phone,round(imrect2rect(train_phone_rects)),sz);

test_cigar_images = multiCrop(conf,test_faces_cigar,round(imrect2rect(test_cigar_rects)),sz);
test_cup_images = multiCrop(conf,test_faces_cup,round(imrect2rect(test_cups_rects)),sz);
test_brush_images = multiCrop(conf,test_faces_brush,round(imrect2rect(test_brush_rects)),sz);
%test_blow_images = multiCrop(conf,test_faces_blow,round(imrect2rect(test_blow_rects)),sz);
test_phone_images = multiCrop(conf,test_faces_phone,round(imrect2rect(test_phone_rects)),sz);

figure,imshow(multiImage(train_cigar_images,false))
figure,imshow(multiImage(train_brush_images,false))
figure,imshow(multiImage(train_cup_images,false))
figure,imshow(multiImage(train_phone_images,false))
figure,imshow(multiImage(test_cigar_images,false))

%locs = boxes2Locs(conf,train_cigar_rects,train_faces_cigar);


toShow = train_faces_cup;
theRects = train_cups_rects;

[ Z,Zind,x,y ] = multiImage(toShow,false);
figure,imshow(Z)
rects_ = imrect2rect(theRects);
for k = 1:length(toShow)
    rects_(k,1:4) = rects_(k,1:4)+[x(k) y(k) x(k) y(k)];
end
hold on;
plotBoxes2(rects_(:,[2 1 4 3]),'color','green','LineWidth',2);



%%
% 
% psix_train_cigar = getBOWFeatures(conf,model2,train_cigar_images,[]);
% psix_test_cigar = getBOWFeatures(conf,model2,test_cigar_images,[]);
% psix_train_cup = getBOWFeatures(conf,model2,train_cup_images,[]);
% psix_test_cup = getBOWFeatures(conf,model2,test_cup_images,[]);
% cur_test_labels

model2 = model;
model2.numSpatialX = [2];
model2.numSpatialY = [2];

psix_train_cigar = [];
psix_test_cigar = [];
psix_train_cup = [];
psix_test_cup = [];
psix_train_brush = [];
psix_test_brush = [];
psix_train_phone = [];
psix_test_phone = [];
psix_train_other = [];
conf.features.vlfeat.cellsize = 8;
% matlabpool

psix_train_cigar = [psix_train_cigar;getBOWFeatures(conf,model2,train_cigar_images,[])];
psix_test_cigar = [psix_test_cigar;getBOWFeatures(conf,model2,test_cigar_images,[])];
psix_train_cup = [psix_train_cup;getBOWFeatures(conf,model2,train_cup_images,[])];
psix_test_cup = [psix_test_cup;getBOWFeatures(conf,model2,test_cup_images,[])];
psix_train_brush = [psix_train_brush;getBOWFeatures(conf,model2,train_brush_images,[])];
psix_test_brush = [psix_test_brush;getBOWFeatures(conf,model2,test_brush_images,[])];
psix_train_phone = [psix_train_phone;getBOWFeatures(conf,model2,train_phone_images,[])];
psix_test_phone = [psix_test_phone;getBOWFeatures(conf,model2,test_phone_images,[])];


% other_train_images = lipImages_train(setdiff(1:length(t_train),[cigarSetTrain;cupSetTrain;brushSetTrain;phoneSetTrain]));
% psix_train_other = [psix_train_other;getBOWFeatures(conf,model2,other_train_images,lipWindowSize)];
% psix_train_other = [psix_train_other ;imageSetFeatures2(conf,other_train_images,true,hogWindowSize)];;


% figure,imshow(multiImage(test_cup_images,false))

psix_train_cigar = [psix_train_cigar;imageSetFeatures2(conf,train_cigar_images,true,hogWindowSize)];
psix_test_cigar = [psix_test_cigar;imageSetFeatures2(conf,test_cigar_images,true,hogWindowSize)];
psix_train_cup =[ psix_train_cup ;imageSetFeatures2(conf,train_cup_images,true,hogWindowSize)];
psix_test_cup = [psix_test_cup ;imageSetFeatures2(conf,test_cup_images,true,hogWindowSize)];
psix_train_brush =[ psix_train_brush ;imageSetFeatures2(conf,train_brush_images,true,hogWindowSize)];
psix_test_brush = [psix_test_brush ;imageSetFeatures2(conf,test_brush_images,true,hogWindowSize)];
psix_train_phone =[ psix_train_phone ;imageSetFeatures2(conf,train_phone_images,true,hogWindowSize)];
psix_test_phone = [psix_test_phone ;imageSetFeatures2(conf,test_phone_images,true,hogWindowSize)];

% psix_train_cigar = [psix_train_cigar;boxCenters(train_cigar_rects_1)'];
% psix_test_cigar = [psix_test_cigar;boxCenters(test_cigar_rects_1)'];
% psix_train_cup =[ psix_train_cup ;boxCenters(train_cup_rects_1)'];
% psix_test_cup = [psix_test_cup ;boxCenters(test_cup_rects_1)'];
% psix_train_brush =[ psix_train_brush ;boxCenters(train_brush_rects_1)'];
% psix_test_brush = [psix_test_brush ;boxCenters(test_brush_rects_1)'];
% psix_train_phone =[ psix_train_phone ;boxCenters(train_phone_rects_1)'];
% psix_test_phone = [psix_test_phone ;boxCenters(test_phone_rects_1)'];
% % 

% size(test_cigar_rects)

train_samples = [psix_train_cigar,psix_train_cup,psix_train_brush,psix_train_phone];%,...
    %psix_train_other];
test_samples = [psix_test_cigar,psix_test_cup,psix_test_brush,psix_test_phone];

cur_train_labels = [1*ones(length(train_cigar_images),1);...
    2*ones(length(train_cup_images),1);...
    3*ones(length(train_brush_images),1);...
    4*ones(length(train_phone_images),1);];%...
    %5*ones(length(other_train_images),1)];

cur_test_labels = [1*ones(length(test_cigar_images),1);...
    2*ones(length(test_cup_images),1);...
    3*ones(length(test_brush_images),1);...
    4*ones(length(test_phone_images),1)];

test_images = [test_cigar_images,test_cup_images,test_brush_images,test_phone_images];

%%
close all

svm_models = {};
names = {'cigar','cup','brush','phone'};
for k = 1:4
% k=1
%     k = 2
    cur_y_train = 2*(cur_train_labels==k)-1;
    cur_y_test = 2*(cur_test_labels==k)-1;
    ss2 = ss;
    ss2 = '-t 0 -c .1 w1 1';
    cur_svm_model= svmtrain(cur_y_train, double(train_samples'),ss2);
    svm_models{k} = cur_svm_model;
    %[predicted_label, ~, cur_model_res] = svmpredict(cur_y_test,double(test_samples'),cur_svm_model);
    
    w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;    
    cur_model_res = w'*test_samples;
    
    %     TT = .1;
    %     cur_model_res = normalise(cur_model_res);
    %     cur_model_res(cur_y_test==1)= cur_model_res(cur_y_test==1).*(rand(1,sum(cur_y_test==1)) >= TT);
    %
    %     plot(cur_svm_model.sv_coef)
    
    [prec,rec,aps] = calc_aps2(cur_model_res',cur_y_test==1);
    figure(1),plot(rec,prec); 
    title([names{k} ' : '  num2str(aps)]);
    [r,ir] = sort(cur_model_res,'descend');
    figure(2),imshow(multiImage(test_images(ir(1:50)),false));   
    title([names{k} ' : '  num2str(aps)]);
    pause;
end
%% now, check how well the classifier performs when given the real 
% lip images, constrained to the above given classes

psix_test_bow = getBOWFeatures(conf,model2,lipImages_test,[]);
psix_test_hog = imageSetFeatures2(conf,lipImages_test,true,[40 40]);

feats_test_total = [psix_test_bow;psix_test_hog]';%;boxCenters(lipBoxes_test_r_2)']';
cur_svm_model =     svm_models{1};
w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;
cur_model_res = feats_test_total*w;

%sel_ = [cigarSetTest;cupSetTest;brushSetTest;phoneSetTest];
sel_ = 1:length(cur_model_res);

[r,ir] = sort(cur_model_res(sel_),'descend');
lipImages_test_reduced = lipImages_test(sel_);
test_faces_2_reduced = test_faces_2(sel_);
figure(2),imshow(multiImage(lipImages_test_reduced(ir(1:50)),false));


[prec,rec,aps] = calc_aps2(cur_model_res,t_test);
plot(rec,prec)
% figure(2),imshow(multiImage(test_faces_2_reduced(ir(1:50)),false));
    
%     figure(2),imshow(multiImage(lipImages_test(t_test),false))
%     figure(2),imshow(multiImage(test_faces(t_test),false))
    

%%

% scanning window....

qq = {};
tt = test_faces_2;
cur_svm_model = svm_models{2};
parfor k = 1:length(tt)
    k
     [ hists,hogs,rects,q ] = bowScan( conf,model2, tt{k});
     
     [~,~,curPred] = svmpredict(zeros(size(rects,1),1),double([hists;hogs;0*boxCenters(rects)']'),cur_svm_model);
     curPred = curPred*cur_svm_model.Label(1);
     qq{k} = curPred;
%      Z = zeros(128);
%      for a = 1:length(unique(q))
%          Z(q==a) = curPred(a);
%      end
%      clf;
%      subplot(2,1,2);
%      imagesc(Z); axis image;colorbar;
%      subplot(2,1,1);
%      imagesc(tt{k}); axis image;
%      pause;
end
%%
qqq = cat(2,qq{:});

[qqq_,iqqq] = max(qqq);

[z,iz] = sort(qqq_,'descend');

imshow(tt{iz(1)})
iqqq(:,iz(1))

figure,imshow(multiImage(tt(iz(1:50))))


%%

    
    %[~, ~, test_cigar_cup] = svmpredict(y_test_cigar_cup,double([psix_test_cup,psix_test_cigar]'),svmModel_bow_cigar_cup);

% end
%%
% 
y_train_cigar_cup = [ones(length(train_cup_images),1);-ones(length(train_cigar_images),1)];
y_test_cigar_cup = [ones(length(test_cup_images),1);-ones(length(test_cigar_images),1)];
ss2 = ss;
ss2 = '-t 0 -c 1 w1 1';
svmModel_bow_cigar_cup= svmtrain(y_train_cigar_cup, double([psix_train_cup,psix_train_cigar]'),ss2);
[~, ~, test_cigar_cup] = svmpredict(y_test_cigar_cup,double([psix_test_cup,psix_test_cigar]'),svmModel_bow_cigar_cup);

[prec,rec,aps] = calc_aps2(test_cigar_cup,y_test_cigar_cup==1);
plot(rec,prec)

%%
% some are not actually face actions but should be ignored nontheless
face_action_classes = [conf.class_enum.DRINKING,...
    conf.class_enum.BLOWING_BUBBLES,...
    conf.class_enum.BRUSHING_TEETH,...
    conf.class_enum.SMOKING,...
    conf.class_enum.PHONING,...
    conf.class_enum.PLAYING_VIOLIN,...
    conf.class_enum.PLAYING_GUITAR,...
    conf.class_enum.APPLAUDING,...
    conf.class_enum.CLIMBING,...
    conf.class_enum.CLEANING_THE_FLOOR,...
    conf.class_enum.TAKING_PHOTOS,...
    conf.class_enum.LOOKING_THROUGH_A_MICROSCOPE,...
    conf.class_enum.LOOKING_THROUGH_A_TELESCOPE];
    
non_action_train = find(~ismember(t_train_all,face_action_classes));

non_action_faces = get_full_image(non_action_train);
action_faces = [train_faces_cigar,train_faces_cup,train_faces_brush,train_faces_phone];

sz1 = [40 40];
[f0,sizes] = imageSetFeatures2(conf,action_faces,true,sz1);
f1 = imageSetFeatures2(conf,non_action_faces,true,sz1);

[ws,b,sv,coeff] = train_classifier(f0,f1);

figure,imagesc(HOGpicture(reshape(ws,sizes{1})))

f2 = imageSetFeatures2(conf,test_faces_2,true,sz1);

f2 = normalize_vec(f2);

q = ws'*f2;
[r,ir] = sort(q(1:1000),'descend');
test_faces__ = test_faces_2(1:1000);
figure,imshow(multiImage(test_faces__(ir(1:100))))

szz= sizes{1};
for k = 1:500
%     f2_1 = reshape(f2(:,k),szz);
    subplot(2,2,1);
    imagesc(test_faces__{ir(k)});axis image;
    subplot(2,2,2);
    
    imagesc(HOGpicture((reshape(ws.*f2(:,ir(k)),szz)))); axis image;title('weighted');
    subplot(2,2,3);
    imagesc(HOGpicture(reshape(f2(:,ir(k)),szz))); axis image;title('features');
    subplot(2,2,4);
    imagesc(HOGpicture(reshape(ws,szz))); axis image;title('w');
    pause;
end



figure,imshow(multiImage(train_faces(non_action_train(1:20)),1:20));

imshow(multiImage(lipImages_train(t_train),false))
%figure,imshow(multiImage(train_faces(non_action_train(401:500)),401:500))
%% now try the same without cropping away stuff from the face, just choosing
% the right faces...
cigarSetTrain = find(ismember(t_train_all,conf.class_enum.SMOKING));
cupSetTrain = find(ismember(t_train_all,conf.class_enum.DRINKING));
brushSetTrain = find(ismember(t_train_all,conf.class_enum.BRUSHING_TEETH));
blowSetTrain = find(ismember(t_train_all,conf.class_enum.BLOWING_BUBBLES));
phoneSetTrain = find(ismember(t_train_all,conf.class_enum.PHONING));

cigarSetTest = find(ismember(t_test_all,conf.class_enum.SMOKING));
cupSetTest = find(ismember(t_test_all,conf.class_enum.DRINKING));
brushSetTest = find(ismember(t_test_all,conf.class_enum.BRUSHING_TEETH));
blowSetTest = find(ismember(t_test_all,conf.class_enum.BLOWING_BUBBLES));
phoneSetTest = find(ismember(t_test_all,conf.class_enum.PHONING));


cigarTrain_w = [1 2 3 4 7 10:13 17:19 21:25 29:34 36 37 40 41 43 44 46 50:54 56 59 61 62 64 66 68 70 71 72 74 75 77];
cupTrain_w = setdiff(1:length(cupSetTrain),[2 41 48]);
brushTrain_w = setdiff(1:length(brushSetTrain),[11 24 41 52 ]);
blowTrain_w = setdiff(1:length(blowSetTrain),[14 22 24 26 29 30 34 36 38 49 52 64 65 67]);
phoneTrain_w = setdiff(1:length(phoneSetTrain),[3 21]);


cigarTest_w = setdiff(1:length(cigarSetTest),[6 13 19 21 22 23 26 31 33 37 40 42 43 45 46 48 50 52 57 60 64 65 68 71 72 76 78 83 85 86 88 89 97]);
cupTest_w = setdiff(1:length(cupSetTest),[23 51 59 79 84 97]);
brushTest_w = setdiff(1:length(brushSetTest),76);
blowTest_w = setdiff(1:length(blowSetTest),[4 14 15 17 24 28 40 51 53 57 58 63 65 69 73 79 81 83 84 86 96 105 108]);
phoneTest_w = 1:length(phoneSetTest);

train_faces_cigar = train_faces(cigarSetTrain(cigarTrain_w));
train_faces_cup = train_faces(cupSetTrain(cupTrain_w));
train_faces_brush = train_faces(brushSetTrain(brushTrain_w));
train_faces_blow = train_faces(blowSetTrain(blowTrain_w));
train_faces_phone = train_faces(phoneSetTrain(phoneTrain_w));

test_faces_cigar = test_faces(cigarSetTest(cigarTest_w));
test_faces_cup = test_faces(cupSetTest(cupTest_w));
test_faces_brush = test_faces(brushSetTest(brushTest_w));
test_faces_blow = test_faces(blowSetTest(blowTest_w));
test_faces_phone = test_faces(phoneSetTest(phoneTest_w));

ttt = [cupSetTest(cupTest_w);cigarSetTest(cigarTest_w)]

model2 = model;
model2.numSpatialX = [2 4];
model2.numSpatialY = [2 4];

sz = [40 40];
conf.features.vlfeat.cellsize = 8;
train_cigar_images = train_faces_cigar;
train_cup_images = train_faces_cup;
train_brush_images = train_faces_brush;
train_phone_images = train_faces_phone;

test_cigar_images = test_faces_cigar;
test_cup_images = test_faces_cup;
test_brush_images = test_faces_brush;
test_phone_images = test_faces_phone;

figure,imshow(multiImage(train_cigar_images,false))
figure,imshow(multiImage(train_brush_images,false))
figure,imshow(multiImage(train_cup_images,false))
figure,imshow(multiImage(train_phone_images,false))
figure,imshow(multiImage(test_cigar_images,false))

%locs = boxes2Locs(conf,train_cigar_rects,train_faces_cigar);


toShow = train_faces_cup;
theRects = train_cups_rects;

[ Z,Zind,x,y ] = multiImage(toShow,false);
figure,imshow(Z)
rects_ = imrect2rect(theRects);
for k = 1:length(toShow)
    rects_(k,1:4) = rects_(k,1:4)+[x(k) y(k) x(k) y(k)];
end
hold on;
plotBoxes2(rects_(:,[2 1 4 3]),'color','green','LineWidth',2);


%%
model2 = model;
model2.numSpatialX = [1 2 4];
model2.numSpatialY = [1 2 4];

psix_train_cigar = [];
psix_test_cigar = [];
psix_train_cup = [];
psix_test_cup = [];
psix_train_brush = [];
psix_test_brush = [];
psix_train_phone = [];
psix_test_phone = [];
psix_train_other = [];
conf.features.vlfeat.cellsize = 8;
% matlabpool

psix_train_cigar = [psix_train_cigar;getBOWFeatures(conf,model2,train_cigar_images,[])];
psix_test_cigar = [psix_test_cigar;getBOWFeatures(conf,model2,test_cigar_images,[])];
psix_train_cup = [psix_train_cup;getBOWFeatures(conf,model2,train_cup_images,[])];
psix_test_cup = [psix_test_cup;getBOWFeatures(conf,model2,test_cup_images,[])];
psix_train_brush = [psix_train_brush;getBOWFeatures(conf,model2,train_brush_images,[])];
psix_test_brush = [psix_test_brush;getBOWFeatures(conf,model2,test_brush_images,[])];
psix_train_phone = [psix_train_phone;getBOWFeatures(conf,model2,train_phone_images,[])];
psix_test_phone = [psix_test_phone;getBOWFeatures(conf,model2,test_phone_images,[])];


other_train_images = lipImages_train(setdiff(1:length(t_train),[cigarSetTrain;cupSetTrain;brushSetTrain;phoneSetTrain]));
psix_train_other = [psix_train_other;getBOWFeatures(conf,model2,other_train_images,[])];
psix_train_other = [psix_train_other ;imageSetFeatures2(conf,other_train_images,true,[80 80])];


% figure,imshow(multiImage(test_cup_images,false))

psix_train_cigar = [psix_train_cigar;imageSetFeatures2(conf,train_cigar_images,true,[])];
psix_test_cigar = [psix_test_cigar;imageSetFeatures2(conf,test_cigar_images,true,[])];
psix_train_cup =[ psix_train_cup ;imageSetFeatures2(conf,train_cup_images,true,[])];
psix_test_cup = [psix_test_cup ;imageSetFeatures2(conf,test_cup_images,true,[])];
psix_train_brush =[ psix_train_brush ;imageSetFeatures2(conf,train_brush_images,true,[])];
psix_test_brush = [psix_test_brush ;imageSetFeatures2(conf,test_brush_images,true,[])];
psix_train_phone =[ psix_train_phone ;imageSetFeatures2(conf,train_phone_images,true,[])];
psix_test_phone = [psix_test_phone ;imageSetFeatures2(conf,test_phone_images,true,[])];

% psix_train_cigar = [psix_train_cigar;boxCenters(train_cigar_rects_1)'];
% psix_test_cigar = [psix_test_cigar;boxCenters(test_cigar_rects_1)'];
% psix_train_cup =[ psix_train_cup ;boxCenters(train_cup_rects_1)'];
% psix_test_cup = [psix_test_cup ;boxCenters(test_cup_rects_1)'];
% psix_train_brush =[ psix_train_brush ;boxCenters(train_brush_rects_1)'];
% psix_test_brush = [psix_test_brush ;boxCenters(test_brush_rects_1)'];
% psix_train_phone =[ psix_train_phone ;boxCenters(train_phone_rects_1)'];
% psix_test_phone = [psix_test_phone ;boxCenters(test_phone_rects_1)'];
% % 

% size(test_cigar_rects)

train_samples = [psix_train_cigar,psix_train_cup,psix_train_brush,psix_train_phone];%,...
    psix_train_other];
test_samples = [psix_test_cigar,psix_test_cup,psix_test_brush,psix_test_phone];

cur_train_labels = [1*ones(length(train_cigar_images),1);...
    2*ones(length(train_cup_images),1);...
    3*ones(length(train_brush_images),1);...
    4*ones(length(train_phone_images),1);];%...
    %5*ones(length(other_train_images),1)];

cur_test_labels = [1*ones(length(test_cigar_images),1);...
    2*ones(length(test_cup_images),1);...
    3*ones(length(test_brush_images),1);...
    4*ones(length(test_phone_images),1)];

test_images = [test_cigar_images,test_cup_images,test_brush_images,test_phone_images];


mImage(train_cup_images)

%%
close all

svm_models = {};
names = {'cigar','cup','brush','phone'};
for k = 1:4
% k=1
%     k = 2
    cur_y_train = 2*(cur_train_labels==k)-1;
    cur_y_test = 2*(cur_test_labels==k)-1;
    ss2 = ss;
    ss2 = '-t 0 -c .1 w1 1';
    cur_svm_model= svmtrain(cur_y_train, double(train_samples'),ss2);
    svm_models{k} = cur_svm_model;
    %[predicted_label, ~, cur_model_res] = svmpredict(cur_y_test,double(test_samples'),cur_svm_model);
    
    w = cur_svm_model.Label(1)*cur_svm_model.SVs'*cur_svm_model.sv_coef;    
    cur_model_res = w'*test_samples;
    
    %     TT = .1;
    %     cur_model_res = normalise(cur_model_res);
    %     cur_model_res(cur_y_test==1)= cur_model_res(cur_y_test==1).*(rand(1,sum(cur_y_test==1)) >= TT);
    %
    %     plot(cur_svm_model.sv_coef)
    
    [prec,rec,aps] = calc_aps2(cur_model_res',cur_y_test==1);
    h = figure(1),plot(rec,prec); 
    title([names{k} ' (full): '  num2str(aps)]);
    saveas(h, fullfile('~/notes/images/refinement/',[names{k} '_full_roc.fig']));
    
    
    [r,ir] = sort(cur_model_res,'descend');
    h = figure(2),imshow(multiImage(test_images(ir(1:50)),false));               
    title([names{k} ' (full): '  num2str(aps)]);
    saveas(h, fullfile('~/notes/images/refinement/',[names{k} '_full_imgs.fig']));
%     pause;
end

%% learn where to look for different classes.
ttt = test_faces_2(t_test);
close all;
[Zs,pts] = createConsistencyMaps(train_cup_rects_1,[128 128],[],inf,[19 7])
figure,imagesc(Zs{1})
figure,imagesc(ttt{24});




% 
% [Zs,pts] = createConsistencyMaps(test_brush_rects_1,[128 128],[],inf,[19 7])
% figure,imagesc(Zs{1})

XX = imageSetFeatures2(conf,train_cup_images(7),true,[]);
x_cluster = makeCluster(XX,[]);
conf.features.winsize = [5 5];
x_cluster = train_patch_classifier(conf,x_cluster,get_full_image(~t_train),'suffix','x_cluster','override',true);
figure,imshow(showHOG(conf,x_cluster))

x_cluster_results = applyToSet(conf,x_cluster,test_faces,[],'x_res_loc','override',true,...
    'rotations',0,'useLocation',Zs);






