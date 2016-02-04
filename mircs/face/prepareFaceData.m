initpath;
config;

conf.class_subset = conf.class_enum.DRINKING;

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

conf.max_image_size = inf;
% prepare the data...

%%

load train_landmarks_full_face.mat; 
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat

dontCareClasses = [conf.class_enum.SMOKING,conf.class_enum.BRUSHING_TEETH,...
    conf.class_enum.BLOWING_BUBBLES];

[~,train_dontcare,~] = getImageSet(conf,'train',1,0,dontCareClasses);
[~,test_dontcare,~] = getImageSet(conf,'test',1,0,dontCareClasses);

% remove the images where no faces were detected.
train_labels=train_labels(train_dets.cluster_locs(:,11));
test_labels=test_labels(test_dets.cluster_locs(:,11));
train_dontcare = train_dontcare(train_dets.cluster_locs(:,11));
test_dontcare = test_dontcare(test_dets.cluster_locs(:,11));
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
figure,imshow(m_train);
m_test = multiImage(test_faces(ir_test(test_labels_sorted)),true);
figure,imshow(m_test);

% % train_face_objects = getObjectnessBoxes(train_faces);
% % save train_face_objects.mat train_face_objects;
% % test_face_objects = getObjectnessBoxes(test_faces);
% % save test_face_objects.mat test_face_objects;

%%
% 
load missingTestRects.mat;

sel_true_train = 1:66;
ii = ir_train(train_labels_sorted);
sel_true_train = ii(sel_true_train);
m_train = multiImage(train_faces(sel_true_train),true);
figure,imshow(m_train);
% find the minimal detection grade
min_train_score = min(train_face_scores(sel_true_train(train_face_scores(sel_true_train) > -1000)));
sel_true_test = 1:91;
ii = ir_test(test_labels_sorted);
sel_true_test = ii(sel_true_test);
m_test = multiImage(test_faces(sel_true_test),true);
figure,imshow(m_test);
% find the minimal detection grade

%min_test_score = min(test_face_scores(sel_true_test));
min_test_score = min_train_score;
t_train = train_labels(train_face_scores>=min_train_score);
test_face_scores(missingTestInds) = min_train_score;
t_test = test_labels(test_face_scores>=min_test_score);
t_train_dontcare = train_dontcare(train_face_scores>=min_train_score);
t_test_dontcare = test_dontcare(test_face_scores>=min_test_score);

% missingTestInds = (test_face_scores < min_train_score) & test_labels';
% test_ids_d_1 = test_ids(test_dets.cluster_locs(:,11));
% missingTestRects = selectSamples(conf,test_ids_d_1(missingTestInds),'missingTestTrueFaces');
% missingTestRects = cat(1,missingTestRects{:});
% missingTestRects(:,4) = missingTestRects(:,4)+missingTestRects(:,2);
% missingTestRects(:,3) = missingTestRects(:,3)+missingTestRects(:,1);
% save missingTestRects.mat missingTestRects missingTestInds
