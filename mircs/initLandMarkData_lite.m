load train_landmarks_full_face.mat;
% AAA = load('train_landmarks_full2.mat');
% train_landmarks_full_face = AAA.train_landmarks_full_ind;
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat;

% remove the images where no faces were detected.
conf.class_subset = conf.class_enum.DRINKING;
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));
all_test_labels=all_test_labels(test_dets.cluster_locs(:,11));
all_train_labels=all_train_labels(train_dets.cluster_locs(:,11));

%[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
%[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
train_ids = train_ids(train_dets.cluster_locs(:,11));
test_ids = test_ids(test_dets.cluster_locs(:,11));

% and do the same for the ids.
%
conf.max_image_size = inf;
[faceLandmarks_train,lipBoxes_train,faceBoxes_train] = landmarks2struct(train_landmarks_full_face,train_ids,train_dets,conf);


t = train_landmarks_full_face;
lengths = zeros(size(t));

    %%

train_face_scores = [faceLandmarks_train.s];

[r_train,ir_train] = sort(train_face_scores,'descend');
[faceLandmarks_test,lipBoxes_test,faceBoxes_test] = landmarks2struct(test_landmarks_full_face,test_ids,test_dets,conf);
test_face_scores = [faceLandmarks_test.s];
[r_test,ir_test] = sort(test_face_scores ,'descend');

train_labels_sorted = train_labels(ir_train);
test_labels_sorted = test_labels(ir_test);
debug_ = 1;


conf.max_image_size = inf;
conf.get_full_image = false;
%%

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);

train_locs_r = train_dets.cluster_locs(train_face_scores>=min_train_score,:);
train_ids_r = train_ids(train_face_scores>=min_train_score,:);
test_locs_r = test_dets.cluster_locs(test_face_scores>=min_test_score,:);
test_ids_r = test_ids(test_face_scores>=min_test_score,:);

%%
close all;

lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);
faceLandmarks_test_t = faceLandmarks_test(test_face_scores>=min_test_score);

lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,[70 70],'both','abs'));
lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,[60 60],'both','abs'));

