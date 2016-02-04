initpath;
config;
load train_landmarks_full_face.mat;
load test_landmarks_full_face.mat;
load newFaceData.mat;
load newFaceData2.mat;

% remove the images where no faces were detected.

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

baseDir = '~/storage/s40Images';
suff= {'.mat','_x2.mat'};
faceLandmarks_train = loadFaces(conf,train_ids,suff,baseDir,'faceLandmarks_train');
faceLandmarks_test = loadFaces(conf,test_ids,suff,baseDir,'faceLandmarks_test');

faceLandmarks_train = processFaceData(conf,train_ids,faceLandmarks_train);
faceLandmarks_test = processFaceData(conf,test_ids,faceLandmarks_test);


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
[faceLandmarks_train,lipBoxes_train,faceBoxes_train] = landmarks2struct(train_landmarks_full_face,train_ids,conf);

t = train_landmarks_full_face;
lengths = zeros(size(t));

%% 
%%
conf.get_full_image = 0;
conf.max_image_size = inf;
for k = 1:length(t)
    k
    tt = t{k};
    if (isempty(tt))
        continue;
    end
%     if (size(tt.xy,1) ~= 39)
%         continue;
%     end
    I = getImage(conf,train_ids{k});
    clf,imshow(I);
    hold on;
    
    curDet = train_dets.cluster_locs(k,:);
    plotBoxes2(curDet([2 1 4 3]));
    
    p0 = curDet([1 2]);
    xy = bsxfun(@plus,p0,boxCenters(tt.xy)/2);
    plot(xy(:,1),xy(:,2),'g.');
    pause;
end
    %%

train_face_scores = [faceLandmarks_train.s];

[r_train,ir_train] = sort(train_face_scores,'descend');
[faceLandmarks_test,lipBoxes_test,faceBoxes_test] = landmarks2struct(test_landmarks_full_face);
test_face_scores = [faceLandmarks_test.s];
[r_test,ir_test] = sort(test_face_scores ,'descend');

train_labels_sorted = train_labels(ir_train);
test_labels_sorted = test_labels(ir_test);
debug_ = 1;

%%

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);



%%
close all;
train_faces = train_faces(train_face_scores>=min_train_score);
get_full_image = get_full_image(train_face_scores>=min_train_score);
% train_faces_4 = train_faces_4(train_face_scores>=min_train_score);

test_faces = test_faces(test_face_scores>=min_test_score);
test_faces_2 = test_faces_2(test_face_scores>=min_test_score);
% test_faces_4 = test_faces_4(test_face_scores>=min_test_score);

lipBoxes_train_r = lipBoxes_train(train_face_scores>=min_train_score,:);
lipBoxes_test_r = lipBoxes_test(test_face_scores>=min_test_score,:);

train_faces_scores_r = train_face_scores(train_face_scores>=min_train_score);
test_faces_scores_r = test_face_scores(test_face_scores>=min_test_score);

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);
faceLandmarks_test_t = faceLandmarks_test(test_face_scores>=min_test_score);

lipBoxes_train_r_2 = round(inflatebbox(lipBoxes_train_r,1.5*[40 40],'both','abs'));

lipImages_train = multiCrop(conf,train_faces,lipBoxes_train_r_2,1.5*[50 50]);

lipBoxes_test_r_2 = round(inflatebbox(lipBoxes_test_r,1.5*[40 40],'both','abs'));

lipImages_test = multiCrop(conf,test_faces,lipBoxes_test_r_2,1.5*[50 50]);

displayRectsOnImages(lipBoxes_train_r_2(t_train,:),train_faces(t_train));
displayRectsOnImages(32+lipBoxes_train_r_2(t_train,:)/2,get_full_image(t_train));


%%
xy_train = {};
for k = 1:length(t_train)
    bc = boxCenters(faceLandmarks_train_t(k).xy);
    %     bc = bc/2+32;
    xy_train{k} = single(bc(:));
end

xy_test = {};
for k = 1:length(t_test)
    bc = boxCenters(faceLandmarks_test_t(k).xy);
    %     bc = bc/2+32;
    xy_test{k} = single(bc(:));
end


