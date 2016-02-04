%%%%%% Experiment 4 %%%%%%%
% Oct. 27, 2013

% using the new face detectors from experiment_0002, detect faces first on
% the train set's drinking images. Afterwards expand to the entire dataset.
% First conclusion : Good! seems to have a better accuracy than anything I've been using so far.
% Now, I shall run this on all of standford40 (with the heuristic that I'll
% take the top half of the image...).  Afterwards, create a datastructure
% to store the landmark localization and face detection results.
initpath;
config;
resultDir = ['~/mircs/experiments/' mfilename];
ensuredir(resultDir);
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);

images = train_ids(train_labels);
conf.get_full_image =false;
for k = 1:length(images)
    I = getImage(conf,images{k});
    images{k} = I(round(1:end/2),:,:);
end
curDir = pwd;
cd /home/amirro/code/3rdparty/voc-release5
L1 = load('models/face_1_final.mat');
L2 = load('models/face_2_final.mat');
L3 = load('models/face_3_final.mat');

models = [L1.model,L2.model,L3.model]
startup;
res = struct('ds',{});
for k = 1:length(images)
    k
    im = images{k};
    
    im = color(im);
    for iModel = 1:length(models)
        [ds, bs] = imgdetect(im, models(iModel),-2);
        top = nms(ds, 0.5);
        ds = ds(top,:);
        res(k,iModel).ds = ds;
    end
end

save ~/code/mircs/face_res.mat res

cd /home/amirro/code/mircs;

subs_ = {};
for k = 1:length(images)
    k
    clf; imagesc(images{k});axis image; hold on;
    dss = [res(k,1).ds(1,:);res(k,2).ds(1,:);res(k,3).ds(1,:)];
    [s,is] = max(dss(:,end));
    dss = dss(is,:);
    plotBoxes2(res(k,1).ds(1,[2 1 4 3]),'r','LineWidth',2);
    plotBoxes2(res(k,2).ds(1,[2 1 4 3]),'g','LineWidth',2);
    plotBoxes2(res(k,3).ds(1,[2 1 4 3]),'b','LineWidth',2);
    plotBoxes2(dss([2 1 4 3]),'m--','LineWidth',2);
    
    subs_{k} = imresize(cropper(images{k},round(dss)),[64 64],'bilinear');
    
    pause;
end

mImage(subs_);

%% already ran on all stanford 40, show some results
subs_ = {};
conf.get_full_image = false;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
T =test_ids(test_labels);
% T = test_ids(randperm(length(test_ids)));
for k = 1:length(T)
    %     k
    %     resFile = fullfile('~/storage/faces_s40_small',strrep(T{k},'.jpg','.mat'));
    %     load(resFile); %-->res
    %
    %     curImage = getImage(conf,T{k});
    %     clf; imagesc(curImage);axis image; hold on;
    %     dss = cat(1,res.ds);
    %     if (isempty(dss))
    %         continue;
    %     end
    %     [s,is] = max(dss(:,end));
    %     dss = dss(is,:);
    %     plotBoxes2(dss([2 1 4 3]),'m--','LineWidth',2);
    
    resFile = fullfile('~/storage/faces_s40',strrep(T{k},'.jpg','.mat'));
    load(resFile); %-->res
    
    curImage = getImage(conf,T{k});
    clf; imagesc(curImage);axis image; hold on;
    dss = cat(1,res.ds);
    if (isempty(dss))
        continue;
        
    end
    [s,is] = max(dss(:,end));
    dss = dss(is,:);
    plotBoxes2(dss([2 1 4 3]),'g--','LineWidth',2);
    drawnow
    pause;
end

%%

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

image_ids = train_ids;
[faceLandmarks,mouth_boxes,face_boxes] = collectResults(conf,image_ids);
cur_set.faceBoxes = face_boxes;
cur_set.lipBoxes = mouth_boxes;
cur_set.faceScores = [faceLandmarks.s];
cur_set.labels = train_labels;
cur_set.imageIDs = image_ids;
cur_set.faceLandmarks = faceLandmarks;
imageData.train = cur_set;
% for k = 1:length(imageData.train.faceLandmarks)
%     k
%     xy = imageData.train.faceLandmarks(k).xy;
%     if (size(xy,1)==39)
%         imageData.train.lipBoxes(k,:) = pts2Box(boxCenters(xy(16:22,:)));
%     end
% end
image_ids = test_ids;
[faceLandmarks,mouth_boxes,face_boxes] = collectResults(conf,image_ids);
cur_set.faceBoxes = face_boxes;
cur_set.lipBoxes = mouth_boxes;
cur_set.faceScores = [faceLandmarks.s];
cur_set.labels = test_labels;
cur_set.imageIDs = image_ids;
cur_set.faceLandmarks = faceLandmarks;
imageData.test = cur_set;
save imageData_new_2.mat imageData;

imageSet = imageData.test;

% crop the images....
conf.get_full_image = false;
cur_set = imageData.train;
rects = zeros(length(cur_set.imageIDs),11);
rects(:,1:4) = cur_set.faceBoxes;
rects(:,11) = 1:length(cur_set.imageIDs);
faces.train_faces = multiCrop(conf,cur_set.imageIDs,rects);

cur_set = imageData.test;
rects = zeros(length(cur_set.imageIDs),11);
rects(:,1:4) = cur_set.faceBoxes;
rects(:,11) = 1:length(cur_set.imageIDs);
faces.test_faces = multiCrop(conf,cur_set.imageIDs,rects);


d = find(cellfun(@(x) ~isempty(find(isnan(x))),faces.test_faces))


mImage(faces.train_faces(imageData.train.labels));

save ~/mircs/experiments/common/faces_cropped_new.mat faces;

subs_ = {};
imageSet = imageData.test;
% scores = imageSet.faceScores;
[s,is] = sort(scores,'descend');
is = 1:length(is);
for k = 1:length(imageSet.imageIDs)
    k
    if (~imageSet.labels(k))
        continue;
    end
    clf; imagesc(getImage(conf,imageSet.imageIDs{k}));axis image; hold on;
    if (~isempty(imageSet.faceLandmarks(k).xy))                
        xy = imageSet.faceLandmarks(k).xy;        
        plotBoxes2( xy(:,[2 1 4 3]),'g');
        
        %         bc = boxCenters(xy);
        %         for kk = 1:size(bc,1)
        %             text(bc(kk,1),bc(kk,2),num2str(kk));
        %         end
        
    end
    pause;
    %     curImage=  getImage(conf,imageSet.imageIDs{is(k)});
    %     subs_{k} = imResample(cropper(curImage,round(imageSet.faceBoxes(is(k),:))),[64 64],'bilinear');
    %     clf; imagesc(subs_{k});axis image; hold on;
end

subs_(cellfun(@isempty,subs_)) = [];
mImage(subs_);




% % showSorted(subs_,[faceLandmarks.s]);
% % %mImage(subs_);
% % bboxes = step(faceDetector,curImage);
% % bboxes = imrect2rect(bboxes);
% % imshow(curImage); hold on; plotBoxes2(bboxes(:,[2 1 4 3]));

% imshow('lena.jpg')

%% show results with the new face detector!!!
%% already ran on all stanford 40, show some results
subs_ = {};
conf.get_full_image = false;
conf.class_subset = conf.class_enum.DRINKING;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
% T =train_ids(train_labels);
% T =test_ids(1:10:end);
T =train_ids;
all_dss = {};
all_crops = {};
% T = T(randperm(length(T)));
toShow = 0;
imgs = {};
for k = 1:length(T)
    
    %     resFile = fullfile('~/storage/faces_s40_small',strrep(T{k},'.jpg','.mat'));
    %     load(resFile); %-->res
    %
    %     curImage = getImage(conf,T{k});
    %     clf; imagesc(curImage);axis image; hold on;
    %     dss = cat(1,res.ds);
    %     if (isempty(dss))
    %         continue;
    %     end
    %     [s,is] = max(dss(:,end));
    %     dss = dss(is,:);
    %     plotBoxes2(dss([2 1 4 3]),'m--','LineWidth',2);
    
    resFile = fullfile('~/storage/faces_s40_big_x2_new',strrep(T{k},'.jpg','.mat'));
    if (~exist(resFile,'file'))
        continue;
    end
    k
    load(resFile); %-->res
    
    curImage = getImage(conf,T{k});
    
    dss = cat(1,res.ds);
    if (isempty(dss))
        continue;
        
    end
%     [s,is] = sort(dss(:,end),'descend');
    dss = esvm_nms(dss,.5);
    
    dss = dss((1:min(size(dss,1),1)),:);
    all_crops{k} = cropper(curImage,round(clip_to_image(dss,curImage)));
    all_dss{k} = dss;
    if (toShow)
        clf; imagesc(curImage);axis image; hold on;
        plotBoxes2(dss(:,[2 1 4 3]),'g--','LineWidth',2);        
        pause
    end    
%         pause;
end
% x = [faces.train_faces(cur_t);all_crops];
% x = x(:);
% mImage(x);
% mImage(all_crops);
all_dss = cat(1,all_dss{:});
myfun = @(x) imResample(x,[80 80],'bilinear');

I1 = showSorted(cellfun2(myfun,all_crops),all_dss(:,end));

% I2 = showSorted(cellfun2(myfun,faces.train_faces(train_labels)),all_dss(:,end));


%% collect all results (top 1 per image) from the newest face detector.

subs_ = {};
conf.get_full_image = false;
conf.class_subset = conf.class_enum.DRINKING;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
T = train_ids;
all_dss = {};
all_crops = {};
toShow = 0;
imgs = {};
for k = 1:length(T)   
    resFile = fullfile('~/storage/faces_s40_big_x2_new',strrep(T{k},'.jpg','.mat'));    
    k
    
    curImage = getImage(conf,T{k});    
%     clf;imagesc(curImage);axis image; pause; continue;
    load(resFile); %-->res    
    dss = cat(1,res.ds);
    
    if (isempty(dss))
        continue;        
    end
    dss = esvm_nms(dss,.5);    
    dss = dss((1:min(size(dss,1),1)),:);
    all_crops{k} = cropper(curImage,round(clip_to_image(dss,curImage)));
    all_dss{k} = dss;
    if (toShow)
        clf; imagesc(curImage);axis image; hold on;
        plotBoxes2(dss(:,[2 1 4 3]),'g--','LineWidth',2);        
        pause
    end    
end

save /home/amirro/mircs/experiments/experiment_0004/dpm_faces_train.mat all_crops all_dss;

%%
T = test_ids;
all_dss = {};
all_crops = {};
toShow = 0;
imgs = {};
for k = 1:length(T)   
    resFile = fullfile('~/storage/faces_s40_big_x2_new',strrep(T{k},'.jpg','.mat'));    
    k
    
    curImage = getImage(conf,T{k});    
%     clf;imagesc(curImage);axis image; pause; continue;
    load(resFile); %-->res    
    dss = cat(1,res.ds);
    
    if (isempty(dss))
        continue;        
    end
    dss = esvm_nms(dss,.5);    
    dss = dss((1:min(size(dss,1),1)),:);
    all_crops{k} = cropper(curImage,round(clip_to_image(dss,curImage)));
    all_dss{k} = dss;
    if (toShow)
        clf; imagesc(curImage);axis image; hold on;
        plotBoxes2(dss(:,[2 1 4 3]),'g--','LineWidth',2);        
        pause
    end    
end


save /home/amirro/mircs/experiments/experiment_0004/dpm_faces_test.mat all_crops all_dss;

% x = [faces.train_faces(cur_t);all_crops];
% x = x(:);
% mImage(x);
% mImage(all_crops);
all_dss = cat(1,all_dss{:});
myfun = @(x) imResample(x,[80 80],'bilinear');

I1 = showSorted(cellfun2(myfun,all_crops),all_dss(:,end));

% I2 = showSorted(cellfun2(myfun,faces.train_faces(train_labels)),all_dss(:,end));
%% and now, from the rotated version...
%% subs_ = {};
conf.get_full_image = false;
conf.class_subset = conf.class_enum.DRINKING;
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
T = train_ids;
all_dss = {};
all_crops = {};
toShow = 1;
imgs = {};
for k = 2200:length(T)   
    k
    resFile = fullfile('~/storage/faces_s40_big_x2_new_rot',strrep(T{k},'.jpg','.mat'));            
    curImage = getImage(conf,T{k});    
    L = load(resFile); %-->res    
    dss = cat(1,L.res.ds);
    if (isempty(dss))
        continue;        
    end
    [~,pick] = esvm_nms(dss(:,1:6),.5);    
    dss = dss(pick,:);
    dss = dss((1:min(size(dss,1),3)),:);
    dss(:,1:4) = dss(:,1:4)/2;
    all_crops{k} = cropper(curImage,round(clip_to_image(dss,curImage)));
    all_dss{k} = dss;
    if (toShow)
        clf; imagesc(curImage);axis image; hold on;
        plotBoxes2(dss(:,[2 1 4 3]),'g--','LineWidth',2);        
        pause
    end
end

save /home/amirro/mircs/experiments/experiment_0004/dpm_faces_train_rot.mat all_crops all_dss;

%%
T = test_ids;
all_dss = {};
all_crops = {};
toShow = 0;
imgs = {};
for k = 1:length(T)   
    resFile = fullfile('~/storage/faces_s40_big_x2_new',strrep(T{k},'.jpg','.mat'));    
    k
    
    curImage = getImage(conf,T{k});    
%     clf;imagesc(curImage);axis image; pause; continue;
    load(resFile); %-->res    
    dss = cat(1,res.ds);
    
    if (isempty(dss))
        continue;        
    end
    dss = esvm_nms(dss,.5);    
    dss = dss((1:min(size(dss,1),1)),:);
    all_crops{k} = cropper(curImage,round(clip_to_image(dss,curImage)));
    all_dss{k} = dss;
    if (toShow)
        clf; imagesc(curImage);axis image; hold on;
        plotBoxes2(dss(:,[2 1 4 3]),'g--','LineWidth',2);        
        pause
    end    
end


save /home/amirro/mircs/experiments/experiment_0004/dpm_faces_test.mat all_crops all_dss;

% x = [faces.train_faces(cur_t);all_crops];
% x = x(:);
% mImage(x);
% mImage(all_crops);
all_dss = cat(1,all_dss{:});
myfun = @(x) imResample(x,[80 80],'bilinear');

I1 = showSorted(cellfun2(myfun,all_crops),all_dss(:,end));



%% 29/4/2014 - yet another version...
[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');

image_ids = train_ids;
[faceLandmarks,mouth_boxes,face_boxes] = collectResults(conf,image_ids);
cur_set.faceBoxes = face_boxes;
cur_set.lipBoxes = mouth_boxes;
cur_set.faceScores = [faceLandmarks.s];
cur_set.labels = train_labels;
cur_set.imageIDs = image_ids;
cur_set.faceLandmarks = faceLandmarks;
imageData.train = cur_set;
% for k = 1:length(imageData.train.faceLandmarks)
%     k
%     xy = imageData.train.faceLandmarks(k).xy;
%     if (size(xy,1)==39)
%         imageData.train.lipBoxes(k,:) = pts2Box(boxCenters(xy(16:22,:)));
%     end
% end
image_ids = test_ids;
[faceLandmarks,mouth_boxes,face_boxes] = collectResults(conf,image_ids);
cur_set.faceBoxes = face_boxes;
cur_set.lipBoxes = mouth_boxes;
cur_set.faceScores = [faceLandmarks.s];
cur_set.labels = test_labels;
cur_set.imageIDs = image_ids;
cur_set.faceLandmarks = faceLandmarks;
imageData.test = cur_set;
save imageData_new_2_big.mat imageData;

imageSet = imageData.test;

% crop the images....
conf.get_full_image = false;
cur_set = imageData.train;
rects = zeros(length(cur_set.imageIDs),11);
rects(:,1:4) = cur_set.faceBoxes;
rects(:,11) = 1:length(cur_set.imageIDs);
%sel_ = [cur_set.faceLandmarks.s] > 0;
sel_ = 1:length(cur_set.imageIDs);
rects__ = rects(sel_,:);rects__(:,11) = 1:size(rects,1);
faces.train_faces = multiCrop(conf,cur_set.imageIDs(sel_),rects__);

cur_set = imageData.test;
rects = zeros(length(cur_set.imageIDs),11);
rects(:,1:4) = cur_set.faceBoxes;
rects(:,11) = 1:length(cur_set.imageIDs);
faces.test_faces = multiCrop(conf,cur_set.imageIDs,rects);


d = find(cellfun(@(x) ~isempty(find(isnan(x))),faces.test_faces))


showSorted(faces.train_faces(imageData.train.labels),imageData.train.faceScores(imageData.train.labels));
showSorted(faces.test_faces(imageData.test.labels),imageData.test.faceScores(imageData.test.labels));

mImage(faces.train_faces(imageData.train.labels));

save ~/mircs/experiments/common/faces_cropped_new.mat faces;

subs_ = {};
imageSet = imageData.train;
scores = imageSet.faceScores;
[s,is] = sort(scores,'descend');
is = 1:length(is);
for k = 1:length(imageSet.imageIDs)
    k
    if (~imageSet.labels(k))
        continue;
    end
    clf; imagesc(getImage(conf,imageSet.imageIDs{k}));axis image; hold on;
    if (~isempty(imageSet.faceLandmarks(k).xy))                
        xy = imageSet.faceLandmarks(k).xy;        
        plotBoxes2( xy(:,[2 1 4 3]),'g');
        
        %         bc = boxCenters(xy);
        %         for kk = 1:size(bc,1)
        %             text(bc(kk,1),bc(kk,2),num2str(kk));
        %         end
        
    end
    pause;
    %     curImage=  getImage(conf,imageSet.imageIDs{is(k)});
    %     subs_{k} = imResample(cropper(curImage,round(imageSet.faceBoxes(is(k),:))),[64 64],'bilinear');
    %     clf; imagesc(subs_{k});axis image; hold on;
end

subs_(cellfun(@isempty,subs_)) = [];
mImage(subs_);

