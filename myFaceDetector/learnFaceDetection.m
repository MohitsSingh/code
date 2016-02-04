% collect positive faces from aflw, coco val images, etc.
% collect negative faces from coco validation set...

addpath(genpath('~/code/utils'));
addpath('/home/amirro/storage/data/VOCdevkit');
addpath('/home/amirro/storage/data/VOCdevkit/VOCcode/');
addpath('/home/amirro/code/3rdparty/edgeBoxes/');
model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
addpath('/home/amirro/code/3rdparty/vlfeat-0.9.18/toolbox');
vl_setup
addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7'));
addpath('~/code/common');
vl_setupnn
net = init_nn_network('imagenet-vgg-s.mat');

model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

% get the fra_db sub images....

fra_db_imgs = {};
for t = 801:900
    I = getImage(conf,fra_db(t).imageID);
    conf.get_full_image = false
    fra_db_imgs{end+1} = cropper(I,round(fra_db(t).raw_faceDetections.boxes(1,:)));
    clf; imagesc2(mImage(fra_db_imgs));
    pause(.1)
end

% re-rank according to fc6, the detection  !!inside!! each image
fra_db_crops = struct('imageIndex',{},'boxes',{},'subImages',{},'feats',{});
for t = 801:900
    t
    I = getImage(conf,fra_db(t).imageID);
    curDetections = fra_db(t).raw_faceDetections.boxes;
    conf.get_full_image = false;
    fra_db_imgs{end+1} = cropper(I,round(fra_db(t).raw_faceDetections.boxes(1,:)));
%     fra_db_imgs{t-800} = cropper(I,round(fra_db(t).raw_faceDetections.boxes(1,:)));
%     clf; imagesc2(mImage(fra_db_imgs));drawnow
%     pause(.1)
%     continue    
    fra_db_crops(t-800).imageIndex = t;
    fra_db_crops(t-800).boxes = curDetections;
    fra_db_crops(t-800).subImages = multiCrop2(I,round(curDetections(:,1:4)));
    feats = extractDNNFeats(fra_db_crops(t-800).subImages,net,16);
    fra_db_crops(t-800).feats = feats.x;
end

re_ranked_crops = {};
for t = 1:length(fra_db_crops)    
    [m,im] = max(w'*normalize_vec(fra_db_crops(t).feats(:,1:2)));
%     [m,im] = max(w'*normalize_vec(fra_db_crops(t).feats));
    re_ranked_crops{t} = fra_db_crops(t).subImages{im};
end
   
% x = extractDNNFeats(fra_db_imgs,net,16);
% x = x.x;

% xx = {}
% for t =1:length(fra_db_imgs)
%     t
%     xx{t} = deepFeatureExtractor.extractFeatures(fra_db_imgs{t});
% end

scores = zeros(100,1);
for t =1:length(fra_db_imgs)    
    scores(t) = fra_db(800+t).raw_faceDetections.boxes(1,end);
end

mImage(fra_db_imgs);title('dpm');
mImage(re_ranked_crops);title('fc6');

showSorted(fra_db_imgs,scores);title('dpm');

x = cat(2,x{:});
x = normalize_vec(x);

showSorted(fra_db_imgs, w'*x); title('fc6');

% save ~/storage/misc/fc6_face_classifier.mat w b

%% set up opts for edgeBoxes (see edgeBoxes.m)
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e4;  % max number of boxes to detect
basePath = '~/storage/mscoco/val2014/COCO_val2014_%012.0f.jpg';
basePath_train = '~/storage/mscoco/train2014/COCO_train2014_%012.0f.jpg';

person_ids = dlmread('~/val_person_ids.txt');
non_person_ids = dlmread('~/val_non_person_ids.txt');
person_paths = {};
for t = 1:length(person_ids)
    curPath = sprintf(basePath,person_ids(t));
    person_paths{t} = curPath;
end
non_person_paths = {};
for t = 1:length(non_person_ids)
    curPath = sprintf(basePath,non_person_ids(t));
    non_person_paths{t} = curPath;
end

non_person_ids_train = dlmread('~/train_non_person_ids.txt');
non_person_paths_train = {};
for t = 1:length(non_person_ids_train)
    curPath = sprintf(basePath_train,non_person_ids_train(t));
    non_person_paths_train{t} = curPath;
end


% displayImageSeries([],non_person_paths,.1)
save ~/non_person_paths.mat non_person_paths
d = person_paths{2};
I = imread(d);
x2(I)

bbs = edgeBoxes2(I,model,opts);
tic, bbs=edgeBoxes(I,model,opts); toc
bbs(:,3) = bbs(:,3)+bbs(:,1);
bbs(:,4) = bbs(:,4)+bbs(:,2);

[pathstr,name,ext] = fileparts(d)
faceDets = load(fullfile('~/storage/mscoco/val_faces',[name '.mat']),'detections');
faceDets = faceDets.detections.boxes;
% 
% % figure(1)
% % for u = 1:size(bbs,1)
% %     clf; imagesc2(I); plotBoxes(bbs(u,:));
% %     pause(.1)
% % end
% x2(I); plotBoxes(bbs(1:1000,:))
% [w h] = BoxSize(bbs);
% bads = min(w./h,h./w) < .5;
% 
% x2(I); plotBoxes(bbs(~bads,:))
% 
% x2(I); U = getSingleRect(1)
% [ovps,ints] = boxesOverlap(bbs,U)
% [m,im] = sort(ovps,'descend');
% for u = 1:5
%     u
%     clf; imagesc2(I); plotBoxes(bbs(im(u),:));pause;
% end
% 
% x2(I); plotBoxes(faceDets)
% 
% for t = 1:length(person_ids)
%     curPath = sprintf(basePath,person_ids(t));
% end
% %
% % I_sub = cropper(I,round(inflatebbox(makeSquare(bbs(im(u),:)),1.5,'both',false)));
% % I_sub = imResample(I_sub,[120 120],'bilinear');
% % [ds, bs] = imgdetect(I_sub, model,-2);
% %
% % top = nms(ds, 0.1);
% % if (isempty(top))
% %     boxes = -inf(1,5);
% % end
% % make
%% load positive,negative face detection from aflw + ms coco
load ~/storage/misc/aflw_crops.mat % ims
% load ~/storage/misc/false_face_det_images.mat; % face_det_imgs, which are false...
feat_pos = extractDNNFeats(ims,net,16);
% feat_neg = extractDNNFeats(face_det_imgs,net,16);
save ~/storage/misc/new_face_feats.mat feat_pos
% now extract fc6 features for all these images...
% requiredMemoryGB = 4*4096*(length(ims)+length(face_det_imgs))/10^6
% negative faces:

all_val_paths = {imgs_and_faces_val.path};
val_neg_sel = ismember(all_val_paths,non_person_paths);
feat_neg = cat(2,imgs_and_faces_val(val_neg_sel).feats);

all_train_paths = {imgs_and_faces_train.path};
train_neg_sel = ismember(all_train_paths,non_person_paths_train);
feat_neg_train = cat(2,imgs_and_faces_train(val_neg_sel).feats);


[x,y] = featsToLabels(feat_pos.x,[feat_neg feat_neg_train]);
x = normalize_vec(x);
figure,plot(y)

[w b]= vl_svmtrain(x(:,:), y, .01,'Verbose','Verbose');

person_sel = find(~val_neg_sel);
bboxes = cat(1,imgs_and_faces_val(person_sel).faces);
img_ids = {};
for t = 1:length(person_sel)
    ims_ids{person_sel(t)} = person_sel(t)*ones(size(imgs_and_faces_val(person_sel(t)).faces,1),1);
end
ims_ids = cat(1,ims_ids{:});

person_sel_feats = normalize_vec(cat(2,imgs_and_faces_val(person_sel).feats));
person_sel_scores_deep = w'*person_sel_feats(:,:)+b;

[u1,iu1] = sort(bboxes(:,end),'descend');
[u2,iu2] = sort(person_sel_scores_deep,'descend');

figure(1);

%%
%%
imgs_1 = {};
imgs_2 = {};
start_ = 2000;
jump_ = floor(500/50);
end_ = length(u1)/5-500;
range_ = start_:jump_:end_;
for t = range_%length(u1)
    t
    k1 = iu1(t); 
    I1 = imread(imgs_and_faces_val(ims_ids(k1)).path);
    k2 = iu2(t); 
    I2 = imread(imgs_and_faces_val(ims_ids(k2)).path);    
    box_inflation_factor = 1.5;
    sub1 = cropper(I1,round(inflatebbox(bboxes(k1,:),box_inflation_factor,'both',false)));
    sub2 = cropper(I2,round(inflatebbox(bboxes(k2,:),box_inflation_factor,'both',false)));
    imgs_1{t} = imResample(sub1,64*[1 1],'bilinear');
    imgs_2{t} = imResample(sub2,64*[1 1],'bilinear');
    continue;
    clf; 
    subplot(1,2,1); imagesc2(sub1); title(sprintf('dpm %f',bboxes(k1,end)));
    subplot(1,2,2); imagesc2(sub2); title(sprintf('deep %f',u2(t)));
    pause
end
%
close all
mImage(imgs_1(range_));title('dpm')
mImage(imgs_2(range_));title('fc6');

%%
imgSize = 64;
opts.maxImageSize = imgSize;
spSize = 50;
opts.pixNumInSP = spSize;
opts.show  =false;     

totalSal = zeros(imgSize);
totalSal_bd = zeros(imgSize);
t = 0;
s = zeros(imgSize,imgSize,3);
figure(1)
% range = 1:5:1000%:length(ims);
for u = range_
%     u
    curIm = imgs_1{u};
    if (length(size(curIm))==2)
        curIm = cat(3,curIm,curIm,curIm);
    end
%     break
    s = s+(imResample(im2double(curIm),[imgSize imgSize],'bilinear'))/length(range_);
    [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(curIm),opts);
    t = t+1;
    totalSal = totalSal+ imResample(sal,size(totalSal),'bilinear');
    totalSal_bd = totalSal_bd+imResample(sal_bd,size(totalSal),'bilinear');
    if (mod(t,10)==0)
        u
        clf;       
        vl_tightsubplot(2,2,1);imagesc2(normalise(totalSal));
        vl_tightsubplot(2,2,2);imagesc2(normalise(-totalSal_bd));
        vl_tightsubplot(2,2,3);imagesc2(normalise(s));
%         [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(normalise(s)),opts);
%         vl_tightsubplot(2,2,4);imagesc2(normalise(sal));
        drawnow
    end
%     pause
end
%%
% nnz(u2>=-.2256)
%% a good threshold is 0.-2256 for thw fc6 detector.
%% 
all
%%

figure,plot(sort(bboxes(:,end)),'r-');
hold on; plot(sort(person_sel_scores_deep),'b-');


%%
