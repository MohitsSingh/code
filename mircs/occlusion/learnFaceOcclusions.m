initpath;
config;
[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');
conf.max_image_size = inf;
conf.class_subset = 9;
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
% sel_true_train_not = [2 52 56:57]; %
% sel_true_train = setdiff(1:55,sel_true_train_not);
sel_true_train = 1:66;
ii = ir_train(train_labels_sorted);
sel_true_train = ii(sel_true_train);
m_train = multiImage(train_faces(sel_true_train),true);
figure,imshow(m_train);
% find the minimal detection grade
% min_train_score = min(train_face_scores(sel_true_train));
min_train_score = -.882;

% min_train_score = -1; % TODO!

% select a true subset for false drinking images
% sel_true_test_not = [60 67 70 72 74]; %
% sel_true_test = setdiff(1:74,sel_true_test_not);
sel_true_test = 1:91;
ii = ir_test(test_labels_sorted);
sel_true_test = ii(sel_true_test);
m_test = multiImage(test_faces(sel_true_test),true);
figure,imshow(m_test);
% find the minimal detection grade
min_test_score = min(test_face_scores(sel_true_test));
% min_test_score = -1; % TODO!
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);

close all;
load('test_face_objects');
load('train_face_objects');

train_face_objects = train_face_objects(train_face_scores>=min_train_score);
train_faces = train_faces(train_face_scores>=min_train_score);

test_face_objects = test_face_objects(test_face_scores>=min_test_score);
test_faces = test_faces(test_face_scores>=min_test_score);

lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,2));
lipImages_train = multiCrop(conf,train_faces,lipBoxes_train_r_2,[50 NaN]);
figure,imshow(multiImage(lipImages_train(t_train),false));

lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,2));
lipImages_test = multiCrop(conf,test_faces,lipBoxes_test_r_2,[50 NaN]);
figure,imshow(multiImage(lipImages_test(t_test),false));

