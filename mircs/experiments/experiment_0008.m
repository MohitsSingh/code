%%%%%% Experiment 8 %%%%%%%
% Nov. 4, 2013
% the purpose of this experiment is to have a better pose estimate
% for discovered faces.

initpath;
config;
addpath('/home/amirro/code/3rdparty/recognitionUsingRegions/util/');
addpath('/home/amirro/code/3rdparty/sliding_segments');
L_imgs = load('/home/amirro/mircs/experiments/common/faceClusters_big_ims.mat');
mkdir('~/mircs/experiments/experiment_0008');
load ~/mircs/experiments/experiment_0008/X_ims.mat X_ims
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
L_train = load('~/mircs/experiments/experiment_0004/dpm_faces_train.mat');
L_test = load('~/mircs/experiments/experiment_0004/dpm_faces_test.mat');

imgs_train = L_train.all_crops;
L_train = rmfield(L_train,'all_crops');
imgs_test = L_test.all_crops;
L_test = rmfield(L_test,'all_crops');

% mImage(imgs_train);
% mImage(imgs_test);
for k = 1:length(imgs_train)
    if (length(size(imgs_train{k})) < 3)
        imgs_train{k} = repmat(imgs_train{k},[1 1 3]);
    end
end

for k = 1:length(imgs_test)
    if (length(size(imgs_test{k})) < 3)
        imgs_test{k} = repmat(imgs_test{k},[1 1 3]);
    end
end

imgs_train = cellfun2(@(x)im2uint8(imResample(im2single(x),[80 80],'bilinear')),imgs_train);
imgs_test = cellfun2(@(x)im2uint8(imResample(im2single(x),[80 80],'bilinear')),imgs_test);

imgs_train = imgs_train(cellfun(@(x)~isempty(x),imgs_train));
imgs_test = imgs_test(cellfun(@(x)~isempty(x),imgs_test));

X_train = (fevalArrays(cat(4,imgs_train{:}),@(x)col(fhog(im2single(x)))));
X_test = (fevalArrays(cat(4,imgs_test{:}),@(x)col(fhog(im2single(x)))));

% try throwing away 9/10 of the images...
% X_ims = X_ims(:,1:5:end);
% L_imgs.ims = L_imgs.ims(1:5:end);
%
conf.get_full_image = false;
knn = 9;
nns_train = zeros(length(imgs_train),knn);
forest = vl_kdtreebuild(X_ims);
wss = zeros(length(imgs_train),3200);
bs = zeros(length(imgs_train),1);
bbs_train = zeros(length(imgs_train),4);
%%
sel_train = find(train_labels);
% X_images,imgs,img_ids,orig_bbs,X_re
for k = 1:length(L_train.all_dss)
    if (isempty(L_train.all_dss{k}))
        L_train.all_dss{k} = [1 1 1 1 0 -10];
    end
end
for k = 1:length(L_test.all_dss)
    if (isempty(L_test.all_dss{k}))
        L_test.all_dss{k} = [1 1 1 1 0 -10];
    end
end

orig_bbs_train = cat(1,L_train.all_dss{:});

% todo - improve by using exact nearest neighbors.
% also, take a subset consisting of the nearest neighbors and expand it by
% looking at the nearest neighbors of the 2nd,3rd,4th (not many) next in
% line nearest neighnors. Then conduct the local search using those nearest
% neighbors, so you wont stray too much from the original. This will also
% be more efficient than running a full knn search at each refinement
% iteration.

knn = 16;
D = l2(X_train',X_ims');
[D,ID] = sort(D,2,'ascend');
D = D(:,1:1000);
ID = ID(:,1:1000);
ID_train_orig = ID;
knn = 9;

profile off;
[nns_train,bbs_train] = matchFaces(conf,X_train,imgs_train,train_ids,orig_bbs_train,X_ims,L_imgs.ims,knn,ID(:,1:100),true);
profile viewer

orig_bbs_test = cat(1,L_test.all_dss{:});
D = l2(X_test',X_ims');
[D,ID] = sort(D,2,'ascend');
D = D(:,1:100);
ID = ID(:,1:100);
ID_test_orig = ID;
[nns_test,bbs_test] = matchFaces(conf,X_test,imgs_test,test_ids,orig_bbs_test,X_ims,L_imgs.ims,knn,ID,false);

save ~/mircs/experiments/experiment_0008/nnData.mat nns_train bbs_train nns_test bbs_test ID_train_orig ID_test_orig

% visualize some of the results.

L_pts = load('/home/amirro/mircs/experiments/experiment_0008/ptsData');
L_pts.ptsData = L_pts.ptsData(1:2:end);
L_pts.poses = L_pts.poses(1:2:end);
L_pts.ellipses= L_pts.ellipses(1:2:end);
% L_pts.ptsData = L_pts.ptsData(1:5:end);
% load the facedata to show it as well.
facesPath = fullfile('~/mircs/experiments/common/faces_cropped_new.mat');
load ~/mircs/experiments/experiment_0008/nnData.mat
L1 = load(facesPath);

% imgs_t = imgs_train(train_labels);
% s_t = orig_bbs_train(train_labels,6);
% showSorted(imgs_t,s_t)

% obtain the mean angle for each of the faces (5nn)
rolls = [L_pts.poses.roll];
pitch = [L_pts.poses.pitch];
yaw = [L_pts.poses.roll];

rolls_ = mean(rolls(ID_train_orig(:,1:5)),2);
pitch_ = mean(pitch(ID_train_orig(:,1:5)),2);
yaw_ = mean(yaw(ID_train_orig(:,1:5)),2);

for k = 1:size(ID_train_orig,1)
    poses_train(k).roll = rolls_(k);
    poses_train(k).pitch = pitch_(k);
    poses_train(k).yaw = yaw_(k);
end


rolls_ = mean(rolls(ID_test_orig(:,1:5)),2);
pitch_ = mean(pitch(ID_test_orig(:,1:5)),2);
yaw_ = mean(yaw(ID_test_orig(:,1:5)),2);

for k = 1:size(ID_test_orig,1)
    poses_test(k).roll = rolls_(k);
    poses_test(k).pitch = pitch_(k);
    poses_test(k).yaw = yaw_(k);
end


load imageData_new;

% another idea - from each angle (yaw), sample 100 random images and find
% most probable angle. or find at 100 nearest neighbor images the most
% probable angle.
% nns_train_1 = nns_train;
% bbs_train = bbs_train;
% nns_train = nns_train_1;
% bbs_train = bbs_train;
tic

for k = 1:length(L_imgs.ims)
    if (toc > 1)
        100*k/length(L_imgs.ims)
        tic
    end
    %     if (abs(yaw(k)) < 50)
    %         continue;
    %     end
    L = load(sprintf('~/storage/landmarks_aflw/%05.0f.mat',k));
    if (isempty(L.landmarks))
        continue;
    end
    landmarks(k) = L.landmarks;
end


%save ~/mircs/experiments/experiment_0008/aflw_landmarks.mat landmarks
load ~/mircs/experiments/experiment_0008/aflw_landmarks.mat landmarks
% landmarks = [];

% refine the mouth locations
faceData_train = showFaceMatchResults(conf,imgs_train,train_ids,L_imgs.ims,nns_train,bbs_train,L_pts,imageData.train,L1.faces.train_faces,landmarks,false);
faceData_test = showFaceMatchResults(conf,imgs_test,test_ids,L_imgs.ims,nns_test,bbs_test,L_pts,imageData.test,L1.faces.test_faces,landmarks,false);



faceData_train.poses = poses_train;
faceData_test.poses = poses_test;
faceData_train.orig_bbs = orig_bbs_train;
faceData_test.orig_bbs = orig_bbs_test;
imageData.train = faceData_train;
imageData.test= faceData_test;
imageData.test.faceScores = orig_bbs_test(:,6);
imageData.train.faceScores = orig_bbs_train(:,6);

for k = 1:4000
    
    if (isempty(faceData_train.faceLandmarks(k).face_seg))
        continue;
    end
    k
    clf;
    displayRegions(im2double(faces.train_faces{k}),{faceData_train.faceLandmarks(k).face_seg},0,-1,1)
    drawnow
    
end

save imageData_new2_exp1_3 imageData;

faces.train_faces = multiCrop(conf,imageData.train.imageIDs,imageData.train.faceBoxes);
faces.test_faces = multiCrop(conf,imageData.test.imageIDs,imageData.test.faceBoxes);

save ~/mircs/experiments/experiment_0008/faces.mat faces


