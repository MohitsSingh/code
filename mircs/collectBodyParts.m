% load ~/storage/misc/s40_face_lm_global.mat; % face_lm_global
% load ~/storage/misc/s40_face_detections.mat; % all_detections
% load ~/storage/misc/s40_frects.mat;% rects
%%
% get face scores...
faceScores = zeros(size(s40_fra));
for u = 1:length(all_detections)
    faceScores(u) = all_detections(u).detections.boxes(1,end);
end
%%
% code relative location of faces vs person bounding boxes
%
% rects = {};
% for t = 1:length(s40_fra)
%     t
%     [ ~,rects{t}] = getImage( conf,s40_fra(t) ,[],true);
% end
%
% rects = cat(1,rects{:});
% save ~/storage/misc/s40_frects.mat rects
toVisualize = false;
all_subs = {};
% all_scores = zeros(size(s40_fra));
for k = 1:length(s40_fra)
    k
    %     k = 1601
    conf.get_full_image = true;
    [I] = getImage(conf,s40_fra(k));
    I_rect = rects(k,:);
    poseData.faceBox = all_detections(k).detections.boxes(1,:);
    curFaceBox = poseData.faceBox(1,1:4)+I_rect([1 2 1 2]);
    if (isinf(curFaceBox(1))),continue,end
    [ovp,int] = boxesOverlap(curFaceBox,I_rect);
    [~,~,b] = BoxSize(curFaceBox);
    int = int /b;
    kp_global = face_lm_global(k).kp_global;
    poseData.landmarks = [boxCenters(kp_global(:,1:4)) kp_global(:,end)];
    % showCoords(poseData.landmarks,[],'color','y');
    faceScale = curFaceBox(4)-curFaceBox(2);
    types = {'eye_left','eye_right','mouth','ear_left','ear_right'};
    indices = [6 19 9 5 18];
    valids = poseData.landmarks(indices,end);
    boxSize = faceScale/2;
    boxes = inflatebbox(poseData.landmarks(indices,[1 2 1 2]),boxSize,'both',true);
    if (toVisualize)
        disp(['face score: ' num2str(poseData.faceBox(end))]);
        disp(['face in person box: ' num2str(int)]);
        clf; imagesc2(I); plotBoxes(curFaceBox);
        plotBoxes(boxes,'m--','LineWidth',2);
        showCoords(boxCenters(boxes),[],'color','y','FontSize',20);
        pause
    end
    curImgs = multiCrop2(I,round(boxes));
    all_subs{k} = curImgs;
end

save ~/storage/misc/all_sub all_subs faceScores
%%
isTrain = [s40_fra.isTrain];
for z = 1:40            
    class_sel = [s40_fra.classID]==z;%conf.class_enum.DRINKING;
    face_score_sel = faceScores > -2;
    img_sel = isTrain & class_sel & face_score_sel;
%     img_sel = class_sel & face_score_sel;
    all_subs1 = cat(1,all_subs{img_sel});
    curScores = faceScores(img_sel);
    [u,iu] = sort(curScores,'descend');
    cur_sub_imgs = all_subs1(iu,3);
    cur_sub_imgs = paintRule(cur_sub_imgs,u>0);
    clf;imagesc2(mImage(cur_sub_imgs));
    title(conf.classes{z});
    clc;
    nnz(img_sel)
    pause
end
% res = struct('result',featuresFromHead(I, poseData, initData.net_s,16,initData.net_deep,33));

% extract deep features from all the mouth regions
networkPath = 'imagenet-vgg-s.mat';
net = init_nn_network();
net
[res] = extractDNNFeats(imgs,net,16,prepareSimple)


%% create an imdb

valids = faceScores > 1;
mouth_subs = all_subs(valids);mouth_subs = cat(1,mouth_subs{:});
mouth_subs = mouth_subs(:,3);
mouth_subs_1 = mouth_subs;
sets = [s40_fra(valids).isTrain];
% jitter the positive examples...
%train_samples = mouth_subs(
mouth_subs = multiResize(mouth_subs,[32 32]);
labels = [s40_fra(valids).classID];
labels(labels~=9) = 2;
labels(labels==9) = 1;
mouth_subs = single(im2uint8(cat(4,mouth_subs{:})));
mean_img = mean(mouth_subs,4);
imdb.images.data = single(mouth_subs);
imdb.images.data_mean = single(mean_img);
imdb.images.labels = single(labels);


% try something else, train a classifier on the activations.

load('data/cifar-baseline/net-epoch-20.mat');
load cifar_data_mean
imgs_train = mouth_subs(:,:,:,sets~=3);
cur_labels = 2*(labels==1)-1;
ff = find(labels_train==1);
net.layers = net.layers(1:11);
feats_train = vl_simplenn(net,bsxfun(@minus,imgs_train,data_mean));
%%
x = squeeze(feats_train(8).x);
x = reshape(x,[],length(labels));
x = normalize_vec(x);
sel_train = sets==1;
labels_train = cur_labels(sets==1);
mouth_test_images = bsxfun(@minus,mouth_subs(:,:,:,sets==0),imdb.images.data_mean);
%%
figure(1);
for lambdas = .01%^[.0001 .01 .1]
    lambdas
    [w b] = vl_svmtrain(x(:,sel_train), labels_train', .01);
    scores_test = w'*x(:,~sel_train);
    labels_test = cur_labels(~sel_train);
    clf;
    vl_pr(labels_test',scores_test');
%     pause
end
%%
figure(1); clf;
[r,ir] = sort(scores_test,'descend');
for q = 1:length(ir)
    q
    k = ir(q);
    clf; imagesc2(uint8(squeeze(mouth_test_images(:,:,:,k))+128));
    pause
end


%%
%%
feats_train = feats_train.x;
% imshow(uint8(imgs_train(:,:,:,ff(2))))

% imdb.images.labels(labels~=9) = 2;
% imdb.images.labels(labels==9) = 1;
f = find(sets==1);
new_sets = single(sets);
new_sets(f(1:3:end)) = 2;
new_sets(sets==0) = 3;
toReplicate = sets==1 & labels == 1;
repFactor = 10;
imgs_to_replicate = imdb.images.data(:,:,:,toReplicate);
% x2(mouth_subs_1(toReplicate));
IJ = jitterImage( imgs_to_replicate, 'maxn',5 ,'hasChn',true,'nTrn',5,'flip',1,'mTrn',5);
jittered = {};
for t = 1:size(IJ,5)
    jittered{end+1} = mat2cell2(IJ(:,:,:,:,t),[1 1 1 10]);
end
jittered = squeeze(cat(1,jittered{:}));
jittered =cat(4,jittered{:});
%U = mat2cell2(uint8(IJ),[1,1,1,90]);x2(U)

imdb.images.data = cat(4,imdb.images.data,jittered);
imdb.images.set = [new_sets ones(1,length(jittered))];
imdb.images.labels = [imdb.images.labels ones(1,length(jittered))];
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = conf.classes;
imdbPath = 'data/cifar-baseline_mouths/imdb.mat';
save(imdbPath,'-struct','imdb');



%% re-calculate features for all of these images...
%cur_sub_imgs

% first, make a cnn_cifar with this
cnn_cifar_2();

R = load('/home/amirro/storage/data/cifar-baseline_mouths/net-epoch-20.mat');
% R.net

mouth_test_images = bsxfun(@minus,mouth_subs(:,:,:,sets==0),imdb.images.data_mean);
net = R.net; net.layers = net.layers(1:end-1);
res = vl_simplenn(net,mouth_test_images);
p = squeeze(res(end).x);
figure,plot(p(2,:));

%%
R = load('/home/amirro/storage/data/cifar-baseline_mouths/net-epoch-20.mat');
% imo = prepareForDNN(imgs,net,prepareSimple);

%mouth_test_images
net = R.net; net.layers = net.layers(1:end-1);
imo = squeeze(mat2cell2(mouth_subs(:,:,:,sets==0)/255,[1,1,1,nnz(sets==0)]));
imo = prepareForDNN(imo,R.net);
res = vl_simplenn(net,imo(:,:,:,:));
clear res
p = squeeze(res(end).x);
figure,plot(p(1,:));

%%
figure(1); clf;
[r,ir] = sort(p(1,:),'descend');
for q = 1:length(ir)
    q
    k = ir(q);
    clf; imagesc2(uint8(squeeze(mouth_test_images(:,:,:,k))+128));
    pause
end

vl_pr(2*(labels(sets==0)==1)-1,p(1,:))

%%
%% % test on *all* images
%mouth_test_images

valids = faceScores > -5;
mouth_subs = all_subs(valids);mouth_subs = cat(1,mouth_subs{:});
mouth_subs = mouth_subs(:,3);
sets = [s40_fra(valids).isTrain];
all_labels = 2*([s40_fra(valids).classID] ==9)-1;
test_labels = all_labels(sets==0);
test_subs = mouth_subs(sets==0);
net = R.net; net.layers = net.layers(1:end-1);
%imo = prepareForDNN(test_subs,R.net);
%res = vl_simplenn(net,imo(:,:,:,:));
res = extractDNNFeats(test_subs,net,20);
p = squeeze(res(end).x);
figure,plot(p(1,:));
vl_pr(test_labels,p(1,:))

%% 

