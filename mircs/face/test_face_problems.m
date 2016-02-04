initpath;
config;
conf.class_subset = 9;
[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');
conf.max_image_size = inf;

% conf.class_subset = 30;% smoking...
% conf.class_subset = 3;% brushing

% prepare the data...
%%
load train_landmarks_full_face.mat;
load test_landmarks_full_face.mat;
load newFaceData.mat;
% load newFaceData2.mat
t_all = train_labels;
conf.class_subset = 2;
[~,t_blowing] = getImageSet(conf,'train');
conf.class_subset = 3;
[~,t_brusing] = getImageSet(conf,'train');
conf.class_subset = 30;
[~,t_smoking] = getImageSet(conf,'train');
% revert to drinking :-)
conf.class_subset = 9;

t_dontcare = t_blowing | t_smoking | t_brusing;

% get the test labels as well...
t_all_test = test_labels;
conf.class_subset = 2;
[~,t_blowing_test ] = getImageSet(conf,'test');
conf.class_subset = 3;
[~,t_brusing_test ] = getImageSet(conf,'test');
conf.class_subset = 30;
[~,t_smoking_test ] = getImageSet(conf,'test');
% revert to drinking :-)
conf.class_subset = 30;
test_dontcare = t_blowing_test | t_smoking_test | t_brusing_test;

% remove the images where no faces were detected.
train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));
t_dontcare = t_dontcare(train_dets.cluster_locs(:,11));
test_dontcare = test_dontcare(test_dets.cluster_locs(:,11));

[faceLandmarks_train,lipBoxes_train,faceBoxes_train] = landmarks2struct(train_landmarks_full_face);
train_face_scores = [faceLandmarks_train.s];
[r_train,ir_train] = sort(train_face_scores,'descend');
[faceLandmarks_test,lipBoxes_test,faceBoxes_test] = landmarks2struct(test_landmarks_full_face);
test_face_scores = [faceLandmarks_test.s];
[r_test,ir_test] = sort(test_face_scores ,'descend');

train_labels_sorted = train_labels(ir_train);
test_labels_sorted = test_labels(ir_test);

m_train = multiImage(train_faces(ir_train(train_labels_sorted)),true);
figure,imshow(m_train);

m_test = multiImage(test_faces(ir_test(test_labels_sorted)),true);
figure,imshow(m_test);


% 
figure,imshow(m_train);
ff = find(train_labels_sorted)
tt = ir_train(ff(62));
figure,imshow(train_faces{tt})
ttt = train_dets.cluster_locs(tt,11);
figure,imshow(getImage(conf,train_ids{ttt}))

clusters = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix','faceDet_new','w1',1);

faces_train_try = applyToSet(conf,clusters,train_ids,[],'faces_train_try','override',false,'visualizeClusters',false);

[train_dets,re_train_try] = combineDetections(faces_train_try);


qq = [];
for k = 1:length(re_train_try)
    qq = [qq;re_train_try(k).cluster_locs(tt,:)];
end
rr = visualizeLocs2_new(conf,train_ids,qq);
figure,imshow(multiImage(rr))

figure,imshow(getImage(conf,train_ids{809}))

% figure,imshow(showHOG(conf,clusters(2)))


I = (getImage(conf,train_ids{809}));
II = (I(1:end/2,:,:));
landmarks = detect_landmarks(conf,{II},1,false);

conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_max_scale = 2;
try_again = applyToSet(conf,clusters,train_ids(809),[],'none','toSave',false);

qq = [];
for k = 1:length(re_train_try)
    qq = [qq;try_again(k).cluster_locs];
end

rr2 = visualizeLocs2_new(conf,train_ids(809),qq);
figure,imshow(multiImage(rr2))


% load faces_train_try.mat 

% % train_face_objects = getObjectnessBoxes(train_faces);
% % save train_face_objects.mat train_face_objects;
% % test_face_objects = getObjectnessBoxes(test_faces);
% % save test_face_objects.mat test_face_objects;

%%
% train_face_objects_2 = getObjectnessBoxes(get_full_image);
% save train_face_objects_2.mat train_face_objects_2;
% test_face_objects_2 = getObjectnessBoxes(test_faces_2);
% save test_face_objects_2.mat test_face_objects_2;

m_test = multiImage(test_faces(ir_test(test_labels_sorted)),true);
figure,imshow(m_test);

% select a true subset of true drinking images.
sel_true_train_not = [2 52 56:57]; %
sel_true_train = setdiff(1:55,sel_true_train_not);
ii = ir_train(train_labels_sorted);
sel_true_train = ii(sel_true_train);
m_train = multiImage(train_faces(sel_true_train),true);
figure,imshow(m_train);
% find the minimal detection grade
min_train_score = min(train_face_scores(sel_true_train));

% min_train_score = -1; % TODO!

% select a true subset for false drinking images
sel_true_test_not = [60 67 70 72 74]; %
sel_true_test = setdiff(1:74,sel_true_test_not);
ii = ir_test(test_labels_sorted);
sel_true_test = ii(sel_true_test);
m_test = multiImage(test_faces(sel_true_test),true);
figure,imshow(m_test);
% find the minimal detection grade
min_test_score = min(test_face_scores(sel_true_test));
% min_test_score = -1; % TODO!
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);

%% run again face landmarks for all images, this time using the top half.
matlabpool;
train_ids_1 = train_ids(train_dets.cluster_locs(:,11));

ff = find(train_labels);


faceLandmarks_train_r = detect_landmarks(conf,train_ids_1(train_labels),1,true);

[faceLandmarks_train_rr,lipBoxes_train_r,faceBoxes_train_r] = landmarks2struct(faceLandmarks_train_r);
faceBoxes_train_r1 = inflatebbox(faceBoxes_train_r,1);
face_imgs_train = multiCrop(conf,train_ids_1(train_labels),round(faceBoxes_train_r));
figure,imshow(multiImage(face_imgs_train))

face_scores_1 = [faceLandmarks_train_rr.s];
[r,ir] = sort(face_scores_1,'descend');
figure,imshow(multiImage(face_imgs_train(ir)))

ff = train_ids_1(train_labels);
imshow(getImage(conf,ff{ir(88)}))

ccc = getImage(conf,ff{ir(88)});


try_landmark = detect_landmarks(conf,{ccc},1,true);

[f1,lip1,face1] = landmarks2struct(try_landmark);;
faceBox1 = inflatebbox(face1,1);
face_imgs_train = multiCrop(conf,train_ids_1(train_labels),round(faceBoxes_train_r));




% % 
% % save faceLandmarks_train_r.mat faceLandmarks_train_r;
% % faceLandmarks_test_r = detect_landmarks(conf,test_ids,1,true);
% % save faceLandmarks_test_r.mat faceLandmarks_test_r;


