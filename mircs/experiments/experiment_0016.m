%%%% experiment 16 %%%%
% Dec. 11 2013

% Train a classifier based only on the appearance on the lips area, while constraining
% the pose and score of the face.


initpath;
config;

% load image data (landmarks, etc)
load ~/storage/misc/imageData_new;
res0012_Dir = '~/mircs/experiments/experiment_0012/';
load(fullfile(res0012_Dir,'lipImagesTrain.mat'),'lipImages_orig');
lipImages_train = cellfun2(@(x) imResample(x,[48 48],'bilinear'),lipImages_orig);
load(fullfile(res0012_Dir,'lipImagesTest.mat'),'lipImages_orig');
lipImages_test = cellfun2(@(x) imResample(x,[48 48],'bilinear'),lipImages_orig);

T_facescore = -.6;
faceScores_train = imageData.train.faceScores;
labels_train = row(imageData.train.labels);
components_train = [imageData.train.faceLandmarks.c];

% This is a small hack to remove 1 positive from the training images since
% it points to the wrong face...
q = find(imageData.train.labels & imageData.train.faceScores' >-.7);
f = q(23);
labels_train(f) = 0;

% computer piotr-dollar standard features.
pChns = chnsCompute();
pChns.shrink = 4;
computeFeatsHelper = @(x) cellfun2(@(y) chnsCompute(y,pChns), x);
getChns = @(y) chnsCompute(y, pChns);
getChns1 = @(z) col(cat(3,z.data{:}));
computeFeats = @(x) getChns1(getChns(x));
feats_train = fevalArrays(cat(4,lipImages_train{:}),computeFeats);
feats_test = fevalArrays(cat(4,lipImages_test{:}),computeFeats);
% training
faceScores_test = imageData.test.faceScores;
labels_test = imageData.test.labels;
components_test = [imageData.test.faceLandmarks.c];

% train on frontal faces
T_facescore =-.5;

sel1 = faceScores_train >= T_facescore;

classifier = train_classifier_pegasos(feats_train(:,sel1 & labels_train),feats_train(:,sel1 & ~labels_train),-1);
%%
res_test = classifier.w(1:end-1)'*feats_test;
T_facescore_test =-.5;
sel_test = faceScores_test >= T_facescore_test;% & abs(components_test-7)<=3;
L_det = load('/home/amirro/mircs/experiments/experiment_0012/lipImagesDetTest.mat');

res_test(~sel_test) = -500;
[prec,rec,aps] = calc_aps2(res_test',imageData.test.labels);

[r,ir] = sort(res_test,'descend');
% plot(cumsum(imageData.test.labels(ir)));
% hold on; plot(r,'-');
%%
showSorted(lipImages_test,res_test,150);

%% result - this could be better, but there are some reasonable candidates. I could do with more training data.