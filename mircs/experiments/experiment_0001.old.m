%%%%%% Experiment 1 %%%%%%%
% Oct. 24, 2013

% The purpose of this experiment is to show that it is useful to lock on to
% salient features in faces if I want to find areas of interaction.

% 1. Compute the saliency for all face images.
% 2. Make a measure of the amount of saliency in different face regions
% 3. show how this measure can prune out none interesting faces.

% modified, 

function res = experiment_0001(conf)
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
saliencyFile = fullfile(resultDir,'face_saliency.mat');
% if (nargin < 1)
%     initpath;s
%     config;
% end

imageData = initImageData;

facesPath = fullfile('~/mircs/experiments/common/faces_cropped.mat');
if (exist(facesPath,'file'))
    load(facesPath);
else
    train_faces = crop_faces(conf,imageData.train);
    test_faces = crop_faces(conf,imageData.test);
    faces.train_faces = train_faces;
    faces.test_faces = test_faces;
    
    
    for k = 1:length(faces.train_faces)
        faces.train_faces{k} = im2uint8(faces.train_faces{k});
    end
    for k = 1:length(faces.test_faces)
        faces.test_faces{k} = im2uint8(faces.test_faces{k});
    end
    
    save(facesPath,'faces');
end

if (exist(saliencyFile,'file'))
    res = load(saliencyFile);
end
% measure the saliency relative to different landmarks.

train_saliency = measureKeyPointSaliency(res.sal_train,imageData.train);
test_saliency = measureKeyPointSaliency(res.sal_test,imageData.test);

save(fullfile(resultDir,'exp_result.mat'),'train_saliency','test_saliency');
return;

% %
% %
% %     T = res.sal_train;
% %     stds =zeros(1,length(T));
% %     means_inside = zeros(1,length(T));
% %     means_outside = zeros(1,length(T));
% %     faceMasks = {};
% %     for k = 1:length(T)
% %         b = im2double(T{k});
% %         k
% %         %M(:,k) = col(imResample(b,sz,'bilinear'));
% %         faceLandmarks = imageData.train.faceLandmarks(k);
% %         faceBox = imageData.train.faceBoxes(k,:);
% %         lipBox = imageData.train.lipBoxes(k,:);
% %         xy_lips = box2Pts(lipBox);
% %         xy = faceLandmarks.xy;
% %         xy = bsxfun(@minus,xy,faceBox([1 2 1 2]));
% %         xy_lips = bsxfun(@minus,xy_lips,faceBox(1:2));
% %         xy = xy*size(b,1)/(faceBox(3)-faceBox(1)+1);
% %         xy_lips = xy_lips*size(b,1)/(faceBox(3)-faceBox(1)+1);
% %         xy_c = boxCenters(xy);
% %         chull = convhull(xy_c);
% %         c_poly = xy_c(chull,:);
% %         face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(b,1),size(b,2));
% %         mouth_mask = poly2mask(xy_lips(:,1),xy_lips(:,2),size(b,1),size(b,2));
% %         stds(k) = std(b(face_mask));
% %         means_inside(k) = mean(b(mouth_mask));
% %         means_outside(k) = mean(b(face_mask & ~mouth_mask));
% %
% %     end
% %     ttt = faces.train_faces;
% %     for k = 1:length(ttt)
% %         tttt{k} = imResample(ttt{k},[128 128],'bilinear');
% %     end
% %
% %     tt1 = stds+(means_inside-means_outside)+ismember([imageData.train.faceLandmarks.c],6:11) + (imageData.train.faceScores>-.7);
% %     save(fullfile(resultDir,'exp_result.mat'),'stds','means_inside','means_outside');
% %     sorted_res = showSorted(tttt,tt1,50);
% %
% %     [prec,rec,aps] = calc_aps2(tt1',imageData.train.labels)
% %
% %     sorted_1 = showSorted(ttt,tt1,50);
% %     imwrite(sorted_1,fullfile(resultDir,'saliency_in_mouth_faces.png'));
% %     sorted_2 = showSorted(T,tt1,50);
% %     imwrite(sorted_2,fullfile(resultDir,'saliency_in_mouth_sals.png'));
% %     showSorted(T,tt,64);
% %     M_pos = M(:,imageData.train.labels);
% %     M_neg = M(:,~imageData.train.labels);
% %     M_neg = M_neg(:,1:1:end);
% %
% % %     m = vl_colsubset(tttt,100);
% % %     conf.detection.params.detect_min_scale = 1;
% % %     conf.features.vlfeat.cellsize = 8;
% % %     conf.detection.params.init_params.sbin = conf.features.vlfeat.cellsize;
% % %     conf.features.winsize = [6 6];
% % %     x = {}
% % %     for k = 1:length(m)
% % %         k
% % %         [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,m{k},.5);
% % %         x{k} = X;
% % %     end
% % %     x = cat(2,x{:});
% % %     [~,C] = kmeans2(x',100);
% % %
% % %     for k = 1:length(tttt)
% % %         k
% % %         if (imageData.train.labels(k))
% % %             continue;
% % %         end
% % %         I = tttt{k};
% % %         [X,uus,vvs,scales,t,boxes ] = allFeatures( conf,I,1);
% % %
% % %         dists = l2(X',C);
% % %         dd = min(dists,[],2);
% % %
% % %         clf; subplot(1,2,1); imagesc(I);axis image;
% % %         Z = zeros(size(I,1),size(I,2));
% % %         bc = round(boxCenters(boxes));
% % %         bc_ = sub2ind2(size(Z),fliplr(bc));
% % %         Z(bc_) = dd;
% % %         Z = imdilate(Z,ones(6));
% % %          subplot(1,2,2);imagesc(Z); axis image;
% % %         pause;
% % %     end
% % %
% % %
% % %     [X,uus,vvs,scales,t,boxes ] = allFeatures(conf,m{1},.5);
% % %
% % %     %TODO : extract several saliency features (in face, out of face, eye area (which might be accidentally salient) etc.)
% % %     % Also "discard" regions strictly outside the face (using gpb,
% % %     % morphology, etc).


%     [ws,b,sv,coeff,svm_model] = train_classifier(M_pos,M_neg,.001,1,2);

%     figure,imagesc(reshape(ws,sz(1),sz(2))); axis image;

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
    faceBox = imageSet.faceBoxes(k,:);
    lipBox = imageSet.lipBoxes(k,:);
    xy_lips = box2Pts(lipBox);
    xy = faceLandmarks.xy;
    xy = bsxfun(@minus,xy,faceBox([1 2 1 2]));
    xy_lips = bsxfun(@minus,xy_lips,faceBox(1:2));
    xy = xy*size(b,1)/(faceBox(3)-faceBox(1)+1);
    xy_lips = xy_lips*size(b,1)/(faceBox(3)-faceBox(1)+1);
    xy_c = boxCenters(xy);
    chull = convhull(xy_c);
    c_poly = xy_c(chull,:);
    face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(b,1),size(b,2));
    mouth_mask = poly2mask(xy_lips(:,1),xy_lips(:,2),size(b,1),size(b,2));
    stds(k) = std(b(face_mask));
    means_inside(k) = mean(b(mouth_mask));
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


