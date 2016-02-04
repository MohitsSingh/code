conf.max_image_size = inf;
conf.clustering.min_cluster_size = 1;

[faceLandmarks_train,allBoxes_train,faceBoxes_train] = landmarks2struct(landmarks_train,train_ids,conf);

faceBoxes_train = inflatebbox(faceBoxes_train,1.3);
faceImagesTrain = multiCrop(conf,train_ids,round(faceBoxes_train/2));
scores = [faceLandmarks_train.s];
[r,ir] = sort(scores,'descend');
figure,imshow(multiImage(faceImagesTrain(ir(1:100))))

save faceImagesTrainData faceImagesTrain scores

faceImagesTrain_r = {};
parfor k = 1:length(faceImagesTrain)
    faceImagesTrain_r{k} = imresize(faceImagesTrain{k},[64 64]);
end

%  figure,plot(r)
sel = find(scores > -.6);
figure,imshow(multiImage(faceImagesTrain_r(sel(1:100))))
conf.features.vlfeat.cellsize = 8;
[posX,sizes] = imageSetFeatures2(conf,faceImagesTrain_r(sel),true);

[C_,IC_] = vl_kmeans(posX,10,'Algorithm','Elkan','NumRepetitions',5);

save faceClusterData posX C_ IC_

clusters = makeClusterImages(faceImagesTrain_r(sel),C_,IC_,posX,'faceClusters_2_kmeans');
conf.features.winsize = sizes{1};
clusters = train_patch_classifier(conf,clusters,getNonPersonIds(VOCopts),'suffix','faceDet_new','w1',1);

conf.detection.params.detect_min_scale = .1;
conf.detection.params.detect_add_flip = 1;
conf.detection.params.detect_levels_per_octave = 8;
conf.clustering.min_cluster_size=1

faces_train_try = applyToSet(conf,clusters,train_ids,[],'faces_train_try','override',false);
save faces_train_try.mat faces_train_try
faces_test_try = applyToSet(conf,clusters,test_ids,[],'faces_test_try','override',false);
save faces_test_try.mat faces_test_try
%%

faces_train_try(1).cluster_locs(:,12)

train_dets = combineDetections(faces_train_try);
test_dets = combineDetections(faces_test_try);

conf.get_full_image = false;
get_full_image = visualizeLocs2_new(conf,train_ids,train_dets.cluster_locs,'height',128,'inflateFactor',2);
test_faces_2 = visualizeLocs2_new(conf,test_ids,test_dets.cluster_locs,'height',128,'inflateFactor',2);



save newFaceData2 get_full_image test_faces_2
save newFaceData train_faces test_faces train_dets test_dets
% a1 = setdiff(1:4000,train_dets(1).cluster_locs(:,11))

train_labels_=train_labels(train_dets.cluster_locs(:,11));
train_faces_true = train_faces(train_labels_);

q = get_full_image(ir);
qq = q(train_labels(ir));
figure,imshow(multiImage(qq(1:5:end)))

cd('/home/amirro/code/3rdparty/objectness-release-v1.5');
 ttt= getObjectnessBoxes(qq(1));
 figure,imshow(qq{1}); 
 hold on;
 plotBoxes2(ttt{1}(:,[2 1 4 3]));

[r,ir] = sort(train_dets.cluster_locs(:,11),'descend');
imshow(multiImage(train_faces_2(ir())))



train_landmarks_full_face= detect_landmarks_p(conf,train_faces_true,1);
inflateFactor = 3;
scaleFactor = 1;
sz = [20 40];
[lipImages_try,faceScores_try] = getLipImages(conf,train_faces_true,train_landmarks_full_face,sz,inflateFactor,scaleFactor);
[r,ir] = sort(faceScores_try,'descend');
figure,imshow(multiImage(lipImages_try(ir(1:50)),false))


train_landmarks_full_face= detect_landmarks(conf,train_faces,1);
save train_landmarks_full_face.mat train_landmarks_full_face;
test_landmarks_full_face = detect_landmarks(conf,test_faces,1);
save test_landmarks_full_face.mat test_landmarks_full_face;

%[newDets,dets] = combineDetections(faces_test_try);


haveDetections_train = train_dets.cluster_locs(:,11);
haveDetections_test = test_dets.cluster_locs(:,11);
[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');

train_labels = train_labels(haveDetections_train);


xy = {};

tlocs = train_dets.cluster_locs(train_dets.cluster_locs(:,12) > -.6,:);
b = boxCenters(tlocs);
parfor k = 1:length(tlocs)
    k
    s = getImage(conf,train_ids{tlocs(k,11)});
    sz_ = size(s);
    bb=boxCenters(tlocs(k,:));
    b(k,:) = [bb(1)/sz_(2) bb(2)/sz_(1)];
end


figure,plot(b(:,1),b(:,2),'r+');
axis image

figure,hist(tlocs(:,8),20)


figure,plot(train_dets.cluster_locs(:,12),train_dets.cluster_locs(:,8),'r+')







%% back to MDF!!
conf.get_full_image = false;
true_dets = find(train_labels(train_dets.cluster_locs(:,11)));
false_dets = find(~train_labels(train_dets.cluster_locs(:,11)));
false_dets = vl_colsubset(false_dets',100,'Uniform')';
true_faces_2 = visualizeLocs2_new(conf,train_ids,train_dets.cluster_locs(true_dets,:),'height',128,'inflateFactor',2);
false_faces_2 = visualizeLocs2_new(conf,train_ids,train_dets.cluster_locs(false_dets,:),'height',128,'inflateFactor',2);
imshow(multiImage(true_faces_2))
imshow(multiImage(false_faces_2))


pos_images = true_faces_2;
neg_images = false_faces_2;
new_mdf

test_faces_2 = visualizeLocs2_new(conf,test_ids,test_dets.cluster_locs,'height',128,'inflateFactor',2);

%%