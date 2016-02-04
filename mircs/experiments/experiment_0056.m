% Experiment 0056, Jan 21 2015
%------------------------------
% train a deep network to tell faces from non faces.
initpath;
config
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
% net = init_nn_network('imagenet-vgg-s.mat');
addpath('/home/amirro/code/myFaceDetector');
basePath = '~/storage/mscoco/val2014/COCO_val2014_%012.0f.jpg';
basePath_train = '~/storage/mscoco/train2014/COCO_train2014_%012.0f.jpg';

% 1. obtain face images and non faces images
% train using matconvnet

load ~/storage/misc/aflw_crops.mat % ims scores

figure,hist(scores)
%length(ims(scores>2.5))

pos_ims = ims(scores>2.5);
pos_ims = cellfun2(@(x) imResample(x,[32 32],'bilinear'), pos_ims);

person_ids = dlmread('~/val_person_ids.txt');
non_person_ids = dlmread('~/val_non_person_ids.txt');
non_person_ids_train = dlmread('~/train_non_person_ids.txt');
person_ids_train = dlmread('~/train_person_ids.txt');

% get paths of non-person images for false face detections.
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


non_person_paths_train = {};
for t = 1:length(non_person_ids_train)
    curPath = sprintf(basePath_train,non_person_ids_train(t));
    non_person_paths_train{t} = curPath;
end

person_paths_train = {};
for t = 1:length(person_ids)
    curPath = sprintf(basePath_train,person_ids_train(t));
    person_paths_train{t} = curPath;
end

false_ims = {};
all_boxes = {};
for t = 1:length(non_person_paths)
    t
    name = non_person_paths{t};
    [pathstr,name,ext] = fileparts(name);
    faceDets = load(fullfile('~/storage/mscoco/val_faces',[name '.mat']),'detections');
    boxes = faceDets.detections.boxes;
    if (any(boxes))
        all_boxes{end+1} = boxes;
        I = imread2(non_person_paths{t});
        u = multiCrop2(I,round(boxes(:,1:4)));
        false_ims{end+1} = cellfun2(@(x) imResample(x,[32 32],'bilinear'),u);
    end
end

% save ~/storage/misc/pos_and_neg_faces.mat pos_ims false_ims

false_ims_train = {};
all_boxes_train = {};
for t = 1:length(non_person_paths_train)
    t
    name = non_person_paths_train{t};
    [pathstr,name,ext] = fileparts(name);
    faceDets = load(fullfile('~/storage/mscoco/train_faces',[name '.mat']),'detections');
    boxes = faceDets.detections.boxes;
    if (any(boxes))
        all_boxes_train{end+1} = boxes;
        I = imread2(non_person_paths_train{t});
        u = multiCrop2(I,round(boxes(:,1:4)));
        false_ims_train{end+1} = cellfun2(@(x) imResample(x,[32 32],'bilinear'),u);
    end
end

% false_ims_train = cat(2,false_ims_train{:});
save ~/storage/misc/pos_and_neg_faces.mat pos_ims false_ims false_ims_train
%% now load images from the positive validation set and rank them...
close all
ims = {};
all_boxes_from_pos = {};
for t = 1:length(person_paths)
    t
    name = person_paths{t};
    [pathstr,name,ext] = fileparts(name);
    faceDets = load(fullfile('~/storage/mscoco/val_faces',[name '.mat']),'detections');
    boxes = faceDets.detections.boxes;
    if (any(boxes))
        all_boxes_from_pos{end+1} = boxes;
        I = imread2(person_paths{t});
        u = multiCrop2(I,round(boxes(:,1:4)));
        ims{end+1} = cellfun2(@(x) imResample(x,[32 32],'bilinear'),u);
        %
        %         clf; imagesc2(I); plotBoxes(boxes);
        %         pause
        % %
    end
end

ims = cat(2,ims{:});
mImage(ims);
all_boxes_from_pos = cat(1,all_boxes_from_pos{:});
[u,iu] = sort(all_boxes_from_pos(:,end),'descend');

%%
%% now load images from the positive training set and rank them...,
% discard faces < .7
close all
ims_train = {};
all_boxes_from_pos_train = {};
for t = 1:length(person_paths_train)
    t
    name = person_paths_train{t};
    [pathstr,name,ext] = fileparts(name);
    faceDets = load(fullfile('~/storage/mscoco/train_faces',[name '.mat']),'detections');
    boxes = faceDets.detections.boxes;
    if (isempty(boxes))
        continue;
    end
    goods = boxes(:,6) > .8;
    boxes = boxes(goods,:);
    if (any(boxes))
        all_boxes_from_pos_train{end+1} = boxes;
        I = imread2(person_paths_train{t});
        u = multiCrop2(I,round(boxes(:,1:4)));
        ims_train{end+1} = cellfun2(@(x) imResample(x,[32 32],'bilinear'),u);
        
        %         clf; imagesc2(mImage(u)); pause
        %
        %         clf; imagesc2(I); plotBoxes(boxes);
        %         pause
        % %
    end
end
%%
ims_train = cat(2,ims_train{:});
mImage(ims_train);
all_boxes_from_pos_train = cat(1,all_boxes_from_pos_train{:});
save ~/storage/misc/faces_from_coco_train.mat ims_train all_boxes_from_pos_train
showSorted(ims_train,all_boxes_from_pos_train(:,6));
[u,iu] = sort(all_boxes_from_pos(:,end),'descend');
%%

% mImage(ims(iu(4000:5500)));
% save ~/storage/misc/faces_from_coco_val.mat ims all_boxes_from_pos
% load ~/storage/misc/pos_and_neg_faces_16_data_mean.mat %dataMean
%%
load faces_32_data_mean
L = load('/home/amirro/code/mircs/data/faces-32_big/net-epoch-6.mat');
net = L.net;
% net = cnn_faces_32;
net.layers = net.layers(1:end-1);
net.normalization.averageImage = dataMean;
net.normalization.border = [0 0];
net.normalization.imageSize = [32 32];
net.normalization.keepAspect = true;
range = 1:20:length(ims);
ims_sub = ims(range);
[res1] = extractDNNFeats(ims_sub,net,12)

% L = load('/home/amirro/code/mircs/data/faces-16/net-epoch-10.mat');
% load ~/storage/misc/pos_and_neg_faces_16_data_mean.mat
% net = L.net;
% % net = cnn_faces_32;
% net.layers = net.layers(1:end-1);
% net.normalization.averageImage = dataMean;
% net.normalization.border = [0 0];
% net.normalization.imageSize = [16 16];
% net.normalization.keepAspect = true;
% range = 1:80:length(ims);
% ims_sub = ims(range);
% [res2] = extractDNNFeats(ims_sub,net,12)


%%

[v,iv] = sort(res1.x(1,:),'descend');
[u,iu] = sort(all_boxes_from_pos(range,end),'descend');
mImage(ims_sub(iv));title('net');
mImage(ims_sub(iu));title('dpm');
%%
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
%% predict locations of outputs facial landmarks
load ~/storage/misc/aflw_with_pts.mat ims pts poses scores inflateFactor resizeFactors;

% a = round(inflatebbox([1 1 128 128],1/1.3,'both',false));
% 1. start a neural net with hog filters...
% 2. use a larger neural net...

% resizeFactors
% ims_small = cellfun2(@(x) condition(length(size(x)==3),x,repmat(x,[1 1 3])),ims);
% figure,plot(sort(sizes(:,1)))
% ims_small = cellfun2(@(x) x(a(1):a(3),a(1):a(3),:),ims_small);

% make a dataset for nose center...
resampler = @(x) imResample(x,[64 64],'bilinear');
% ims_small = cellfun2(resampler,ims);
requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
all_kps = squeeze(getKPCoordinates_2(pts,{'NoseCenter'}))+1;

my_kps = squeeze(getKPCoordinates_2(pts,requiredKeypoints))+1;

sizes = cellfun2(@size,ims);
sizes = cat(1,sizes{:});
scaleFactors = 32./sizes(:,1);
all_poses = poses;
all_yaws = [all_poses.yaw];
curScores = row(scores);
%

% im_sel = 1:length(all_yaws);
im_sel = find(abs(all_yaws)<.4 & curScores >= 0);
im_sel = im_sel(1:1:end);
im_sel = intersect(im_sel,find(~any(isnan(all_kps),2)));
%%
% figure(1);
%%
all_ims_small = ims(im_sel);
output_kp = all_kps(im_sel,:);
output_kp_2 = all_kps(im_sel,:,:);

imgSize = 224;
% % save('ims_small.mat','ims_small');
for t = 1:1:length(all_ims_small)
    if (mod(t,100)==0)
        t/length(all_ims_small)
    end
    I = all_ims_small{t};
    sz = fliplr(size2(I));
    a = round(inflatebbox([1 1 sz],1/1.3,'both',false));
    I = cropper(I,a);
    %     clf; imagesc2(I); %plotBoxes(a);
    %     plotPolygons(1+output_kp(t,:)-a(1,1:2),'g+');
    %     pause
    resizeFactor = imgSize/size(I,1);
    all_ims_small{t} = imResample(I,[imgSize imgSize],'bilinear');
    output_kp(t,:) = resizeFactor*(1+output_kp(t,:)-a(1,1:2));
    %     clf; imagesc2(all_ims_small{t}); %plotBoxes(a);
    %     plotPolygons(output_kp(t,:),'r+')
    %     drawnow
    %     pause
end
%%
train_imgs = all_ims_small(1:3:end);
target_xy = output_kp(1:3:end,:);

for u = 1:length(train_imgs)
    figure(1);clf; imagesc2(train_imgs{u});
    plotPolygons(target_xy(u,:),'r+');
    
end

% save aa aa
addpath('~/code/greg/');
X_train = aa(:,1:3:end);
sizes = cell2mat(cellfun2(@size2,train_imgs'));% height/width
factors = sizes(:,1);
%%

X_toTrain = normalize_vec(X_train);
[ model ] = train_kp_regressor(X_toTrain,target_xy./repmat(factors,1,2));
predicted_points = [factors factors].*apply_kp_regressor(X_toTrain,model)';
disp(['mean error: ' num2str(mean( sum( (predicted_points-target_xy).^2,2).^.5))]);

%%
%%
for t = 1:length(train_imgs)
    figure(1);clf; imagesc2(train_imgs{t});
    plotPolygons(target_xy(t,:),'g+');
    plotPolygons(predicted_points(t,:),'r+');
    drawnow;pause
end

%%
test_imgs = all_ims_small(2:3:end);
sizes = cell2mat(cellfun2(@size2,test_imgs'));% height/width
test_factors = sizes(:,1);
target_xy_test = output_kp(2:3:end,:);
X_test = normalize_vec(aa(:,2:3:end));
predicted_points_test = [test_factors test_factors].*apply_kp_regressor(X_test,model)';

for t = 1:length(test_imgs)
    figure(1);clf; imagesc2(test_imgs{t});
    plotPolygons(target_xy_test(t,:),'g+');
    plotPolygons(predicted_points_test(t,:),'r+');
    drawnow;pause
end


%%

addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
load('/home/amirro/code/mircs/data/cifar-baseline/net-epoch-18.mat'); %net
load /home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples/cifar_data_mean.mat; % data_mean

% save ~/code/mircs/train_data.mat train_imgs target_xy
cnn_landmarks_start_cifar;

load('/home/amirro/code/mircs/data/face_landmarks_cifar/net-epoch-200.mat');
net.layers = net.layers(1:end-1);

load /home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples/cifar_data_mean.mat
train_imgs_1 = bsxfun(@minus,single(cat(4,train_imgs{:})),data_mean);

[res] = vl_simplenn(net,train_imgs_1);x = squeeze(res(end).x)';
% plotPolygons(squeeze(res(2).x)','r.')

for t = 1:length(train_imgs)
    figure(1);clf; imagesc2(train_imgs{t});
    plotPolygons(32*x(t,:),'r+');
    plotPolygons(target_xy(t,:),'g+');
    pause;
end

%%
net = init_nn_network;net.layers = net.layers(1:15);
feats_train  = extractDNNFeats(train_imgs,net,16);
%%

test_imgs = all_ims_small(2:3:end);
target_xy_test = output_kp(2:3:end,:);
test_imgs_1 = getImageStackHOG(test_imgs,[64 64],false);

% net = init_nn_network;
% res = extractDNNFeats_parallel(all_ims_small,net);

[res] = vl_simplenn(net,cat(4,test_imgs_1{:}));x = squeeze(res(2).x)';
% plotPolygons(squeeze(res(2).x)','r.')

for t = 1:length(test_imgs)
    clf; imagesc2(test_imgs{t});
    plotPolygons(64*x(t,:),'g+');
    pause;
end





[res] = vl_simplenn(net,cat(4,test_imgs_1{:}));x = squeeze(res(2).x)';
% plotPolygons(squeeze(res(2).x)','r.')

for t = 1:length(test_imgs)
    clf; imagesc2(test_imgs{t});
    plotPolygons(64*x(t,:),'g+');
    pause;
end



goods = ~any(isnan(target_xy),2);
train_imgs = train_imgs(goods);
target_xy = target_xy(goods,:);

for t = 1:length(train_imgs)
    clf; imagesc2(train_imgs{t});
    plotPolygons(target_xy(t,:),'g+');
    pause;
end

save('train_data.mat','train_imgs','target_xy');
cnn_landmarks_start_cifar
A = load('/home/amirro/code/mircs/data/face_landmarks_cifar/net-epoch-10.mat');
net = A.net;net.layers = net.layers(1:end-1);

% load /home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples/cifar_data_mean.mat

% apply to some images...
%im_sel1 = find(abs(all_yaws) < .1 & all_yaws < .2));
% im_sel1 = im_sel(2:8:end);
test_imgs = all_ims_small(2:2:end);
test_imgs_input = fevalArrays(single(cat(4,test_imgs{:})), @fhog2);
dnn_res = vl_simplenn(net, gpuArray(test_imgs_input));
xy = squeeze(dnn_res(end).x)*32;
xy = gather(xy);
for t = 1:length(test_imgs)
    clf; imagesc2(test_imgs{t});
    plotPolygons(xy(:,t)','g+');
    pause;
end
[res] = extractDNNFeats(test_imgs,net,12)
% for t = 1:length(im_sel)
%     k = im_sel(t);
%     clf; imagesc(ims_small{k});
%     plotPolygons(all_kps(k,:)/4,'r+');
%     pause
% end

%%
load ~/storage/misc/faces_from_coco_val.mat ims all_boxes_from_pos
% %
% % t =7
% % % for t = 1:30
t = 8
L = load(['/home/amirro/code/mircs/data/faces-32_big_with_negatives/net-epoch-' num2str(t) '.mat']);
info = L.info;
figure,plot(info.train.error)
hold on; plot(info.val.error,'r');
% end
% infos = [infos{:}];
% trains = [infos.train];
% figure,plot([trains.error])
net = L.net;
% net = cnn_faces_32;
net.layers = net.layers(1:end-1);
net.normalization.averageImage = dataMean;
net.normalization.border = [0 0];
net.normalization.imageSize = [32 32];
net.normalization.keepAspect = true;
range = 1:5:length(ims);
ims_sub = ims(range);
[res1] = extractDNNFeats(ims_sub,net,12);
% re-ranking
%%
[u,iu] = sort(res1.x(1,:),'descend');
mImage(ims_sub(iu(1:40:end/3)));title('deep');

[u,iu] = sort(all_boxes_from_pos(range,6),'descend');


[u,iu] = sort(all_boxes_from_pos(:,6),'descend');
%%
% conf,ims,delay,trues,displayTrue,indexToShow)
jump_ =100;
% add to positive samples all scores > .6
min_score = 0.6;
for t = 5500:jump_:length(u)
    clf; imagesc2(imResample(ims{iu(t)},4,'bilinear'));
    num2str([t u(t)])
    if (u(t) < min_score)
        break
    end
    pause
end



%displayImageSeries(conf,ims_sub(iu),u,0,true(size(ims_sub)),false)

%mImage(ims_sub(iu(1:10:end/5)));title('dpm')



