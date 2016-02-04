%%%%%% Experiment 1 %%%%%%%
% Oct. 24, 2013

% The purpose of this experiment is to show that it is useful to lock on to
% salient features in faces if I want to find areas of interaction.

% 1. Compute the saliency for all face images.
% 2. Make a measure of the amount of saliency in different face regions
% 3. show how this measure can prune out none interesting faces.

% modified - october 29, after having run a much improved face detector.

function res = experiment_0001(conf)
resultDir = ['~/mircs/experiments/' mfilename '_improved'];
ensuredir(resultDir);
saliencyFile = fullfile(resultDir,'face_saliency.mat');
% if (nargin < 1)
%     initpath;s
%     config;
% end

%imageData = initImageData;
load imageData_new;
load ~/mircs/experiments/common/faces_cropped_new.mat;
load '/home/amirro/mircs/experiments/experiment_0001/sals_new.mat';
% load ~/mircs/experiments/common/faces_cropped_new_large.mat;


% imshow(faces_large.train_faces{1});


if (exist(saliencyFile,'file'))
    res = load(saliencyFile);
end
% measure the saliency relative to different landmarks.

train_saliency = measureKeyPointSaliency(sal_train,imageData.train);
test_saliency = measureKeyPointSaliency(sal_test,imageData.test);

save(fullfile(resultDir,'exp_result_new.mat'),'train_saliency','test_saliency');
return;


M_test = zeros(prod(sz),length(res.sal_test));
for k = 1:length(res.sal_test)
    b = im2double(res.sal_test{k});
    M_test(:,k) = col(imResample(b,sz,'bilinear'));
end

[~,~,scores] =svmpredict(zeros(length(res.sal_test),1), M_test',svm_model);
showSorted(faces.test_faces,ismember([imageData.test.faceLandmarks.c],6:11)+scores',100);

%ws'*M_test


for k = 1:length(tt)
    b = T{k};
    %b = b(round(.25*end):round(.75*end),round(.25*end):round(.75*end));
    %b = b(round(.7*end):round(.8*end),round(.4*end):round(.6*end));
    b = imResample(im2double(b),[80 80],'bilinear').*(M_drink-M_all);
    b=  M_drink-M_all;
    b = b(20:60,20:60);
    tt(k) = mean(b(:));
end





%     [s,is] = sort(tt,'descend');
rq = randperm(length(tt));
showSorted(faces.train_faces(rq),tt(rq),50);
showSorted(T(rq),tt(rq),64);
showSorted(faces.train_faces,imageData.train.labels,64);
%
end

% end

function stats = measureKeyPointSaliency(T,imageSet);
stds =zeros(1,length(T));
means_inside = zeros(1,length(T));
means_outside = zeros(1,length(T));
for k = 1:length(T)
    b = im2double(T{k});
    k
    faceLandmarks = imageSet.faceLandmarks(k);
    
    if (isempty(faceLandmarks.xy))
        continue;
    end
    
    faceBox = imageSet.faceBoxes(k,:);
    lipBox = imageSet.lipBoxes(k,:);
    xy_lips = box2Pts(lipBox);
    
    xy = faceLandmarks.xy;
    xy = bsxfun(@minus,xy,faceBox([1 2 1 2]));
    xy_lips = bsxfun(@minus,xy_lips,faceBox(1:2));
    
    sz = size(b);
    b = b(round((sz(1)/6):sz(1)*(5/6)),round((sz(2)/6):sz(2)*(5/6)));
    
    xy = xy*size(b,1)/(faceBox(3)-faceBox(1)+1);    
    xy_lips = xy_lips*size(b,1)/(faceBox(3)-faceBox(1)+1);
    xy_c = boxCenters(xy);
    chull = convhull(xy_c);
    c_poly = xy_c(chull,:);
    
    
    
    
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(b,1),size(b,2));
    mouth_mask = poly2mask(xy_lips(:,1),xy_lips(:,2),size(b,1),size(b,2));
    stds(k) = std(b(face_mask));
    means_inside(k) = mean(b(mouth_mask & face_mask));
    means_outside(k) = mean(b(face_mask & ~mouth_mask));
end
stats.stds = stds;
stats.means_inside = means_inside;
stats.means_outside = means_outside;
end

function F = crop_faces(conf,imageSet)
conf.get_full_image = true;
imageIDS = imageSet.imageIDs;
% crop to a given size.
%         sz = [128 64];
F = {};
tic;
for k = 1:length(imageIDS)
    if (toc > 1)
        tic;
        disp(100*k/length(imageIDS))
    end
    I = getImage(conf,imageIDS{k});
    bbox =  clip_to_image(round(imageSet.faceBoxes(k,:)),I);
    F{k} = cropper(I,bbox);
end
end


