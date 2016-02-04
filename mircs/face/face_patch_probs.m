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
% load newFaceData4.mat;
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

min_train_score  =-.882;
min_test_score = min_train_score;
% min_test_score = -1;
t_train = train_labels(train_face_scores>=min_train_score);
t_test = test_labels(test_face_scores>=min_test_score);
t_test_all = all_test_labels(test_face_scores>=min_test_score);
t_train_all = all_train_labels(train_face_scores>=min_train_score);

faceLandmarks_train_t = faceLandmarks_train(train_face_scores>=min_train_score);
faceLandmarks_test_t = faceLandmarks_test(test_face_scores>=min_test_score);
%%
close all;
train_faces = train_faces(train_face_scores>=min_train_score);
get_full_image = get_full_image(train_face_scores>=min_train_score);
% train_faces_4 = train_faces_4(train_face_scores>=min_train_score);


test_faces = test_faces(test_face_scores>=min_test_score);
test_faces_2 = test_faces_2(test_face_scores>=min_test_score);
% test_faces_4 = test_faces_4(test_face_scores>=min_test_score);

%%
xy_train = {};
for k = 1:length(t_train)
    bc = boxCenters(faceLandmarks_train_t(k).xy);
    %     bc = bc/2+32;
    xy_train{k} = bc(:);
end

lengths_train = cellfun(@length,xy_train);

lipReader;
allDrinkingInds = [cupInds_1 cupInds_2 cupInds_3 strawInds bottleInds];
t_train = train_labels(train_face_scores>=min_train_score);
ff = find(t_train);
ff = ff(allDrinkingInds);
t_train = false(size(t_train));
t_train(ff) = true;
mImage(train_faces(t_train));

close all;

tt_train = lengths_train == 136;
t_train_all_tt = t_train_all(tt_train);
t_train_tt = t_train(tt_train);
train_faces_tt = get_full_image(tt_train);
train_faces_tt_x = train_faces(tt_train);
faceLandmarks_train_tt = faceLandmarks_train_t(tt_train);
mImage(train_faces_tt(t_train_tt));


%%
%%
xy_test = {};
for k = 1:length(t_test)
    bc = boxCenters(faceLandmarks_test_t(k).xy);
    xy_test{k} = bc(:);
end

lengths_test = cellfun(@length,xy_test);

tt_test = lengths_test == 136;
t_test_tt = t_test(tt_test );
t_test_all_tt = t_test_all(tt_test);
test_faces_tt = test_faces_2(tt_test );
test_faces_tt_x = test_faces(tt_test );
faceLandmarks_test_tt = faceLandmarks_test_t(tt_test );
mImage(test_faces_tt(t_test_tt));

xy_test = cat(2,xy_test{tt_test });

% choose the nearest neighbors only from the drinking faces.
% xx_test = imageSetFeatures2(conf,test_faces_tt_x,true,[80 80]);


allPatches_train = {};
for k = 1:length(train_faces_tt)
    k
    I = im2single(rgb2gray(train_faces_tt{k}));
    xy = boxCenters(faceLandmarks_train_tt(k).xy);
    xy = xy/2+32;    
    xy = boxCenters(faceLandmarks_test_tt(k).xy);
    xy = xy/2+32;    
    xy = inflatebbox([xy xy],15,'both',true);    
    a = multiCrop(conf,{I},round(xy),[60 60]);    
    allPatches_train{k} = single(imageSetFeatures2(conf,a,true,[]));
end

allPatches_test = {};
for k = 1:length(test_faces_tt)
    k
     I = im2single(rgb2gray(test_faces_tt{k}));
    xy = boxCenters(faceLandmarks_test_tt(k).xy);
    xy = xy/2+32;    
    xy = boxCenters(faceLandmarks_test_tt(k).xy);
    xy = xy/2+32;    
    xy = inflatebbox([xy xy],15,'both',true);    
    a = multiCrop(conf,{I},round(xy),[60 60]);    
    allPatches_test{k} = single(imageSetFeatures2(conf,a,true,[]));
end

nPatchesPerFace = 68;

% patchProbs = zeros(nPatchesPerFace,length(test_faces_tt));
allPatches_b_train = {};
patchProbs_train = zeros(nPatchesPerFace,length(train_faces_tt));

for k = 1:length(train_faces_tt)
    c = allPatches_train{k};
    for kk = 1:size(c,2)
        allPatches_b_train{kk,k} = c(:,kk);
    end
end



for k = 1:length(test_faces_tt)
    c = allPatches_test{k};
    for kk = 1:size(c,2)
        allPatches_b_test{kk,k} = c(:,kk);
    end
end
for k = 1:nPatchesPerFace
    k
    curPatches = cat(2,allPatches_b_test{k,:});
    curTrainPatches = cat(2,allPatches_b_train{k,non_action_train});
        
    curDists = l2(double(curPatches)',double(curTrainPatches)');
    ccc = exp(-curDists/10);    
    ccc = sum(ccc,2);
    ccc = ccc/max(ccc);
    patchProbs(k,:) = ccc;
    
    curPatches = cat(2,allPatches_b_train{k,:});
    curDists = l2(double(curPatches)',double(curTrainPatches)');
    ccc = exp(-curDists/10);    
    ccc = sum(ccc,2);
    ccc = ccc/max(ccc);
    patchProbs_train(k,:) = ccc;   
    
    
    
%     figure,imagesc(ccc)
%     hist(ccc(:),100)
end


