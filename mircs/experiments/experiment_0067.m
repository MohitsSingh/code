
%% Experiment 0067 %%%%%
%% 26/5/2015
% Make my own facial landmark detector, using regression from cnn features.
if (~exist('initialized','var'))
    cd ~/code/mircs;
    initpath;
    config;
    conf.get_full_image = true;
    wSize = 64;    
    wSize = 32;        
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');
    vl_setupnn;
    initialized = true;
    load ~/storage/misc/aflw_with_pts.mat ims pts poses scores inflateFactor resizeFactors;
end
% try to make a prediction for each keypoint....
%imgs1 = curImgs(1:5:end);
% imgs1 = curImgs(1:5:end);
%featureExtractor = DeepFeatureExtractor(conf,true,16);
% featureExtractor = DeepFeatureExtractor(conf,true,12); % 16->9216, 17->4096 , 12 (conv4)
%featureExtractor = DeepFeatureExtractor(conf,true,18); % very deep fc6

%featureExtractor = DeepFeatureExtractor(conf,true,1+[14 17 19 21]); % conv5,conv6,conv7,fc8
featureExtractor = DeepFeatureExtractor(conf,true,1+14);

F = featureExtractor.extractFeaturesMulti(ims(1),false);size(F)
% F_1 = normalize_vec(F);
showNN(F,imgs1);

im_subset = 1:length(ims);
requiredKeypoints = unique(cat(1,pts.pointNames));
%all_kps = getKPCoordinates_2(ptsData(im_subset),requiredKeypoints)+1;

[r,ir] = sort(scores,'ascend');

im_subset = 1:10:length(ims);

curImgs = ims(im_subset);
all_kps = getKPCoordinates_2(pts(im_subset),requiredKeypoints)+1;
%
%%
for it = 1:50:length(curImgs)
    r(it)
    t = ir(it)
    I = curImgs{t};
%     I = imResample(I,[256 200],'bilinear');
    %clf; imagesc2(curImgs{t});
    clf; imagesc2(I);
    plotPolygons(squeeze(all_kps(t,:,:))+1,'g+');
    pause(.1)
end

%%
imgs = {};
kps = {};
%for t = 1:50:length(curImgs)    
for t = 1:length(curImgs)    
    t
    I = curImgs{t};
%     I = imResample(I,[256 200],'bilinear');
    %clf; imagesc2(curImgs{t});
    resizeFactor = 64/size(I,1);    
    I = imResample(I,[64 64]);
    curPts = squeeze(all_kps(t,:,:))+1;
    curPts = curPts*resizeFactor;
%     clf; imagesc2(I);
%     plotPolygons(curPts,'g+');
%     pause
    imgs{t} = I;
    kps{t} = curPts;
end


save ~/storage/misc/aflw_with_pts_small imgs kps scores poses

%%

%%

sel_test = vl_colsubset(1:length(curImgs),round(.8*length(curImgs)));
sel_train = setdiff(1:length(curImgs),sel_test);
%%
feats_l2 = featureExtractor.extractFeaturesMulti(curImgs,false);
feats_l2_bkp = feats_l2;
% feats_l2 = getImageStackHOG(curImgs);
%%
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
%%
% feats_orig = feats_l2;
requiredKeypoints = requiredKeypoints(9); % just mouth elements;
%req_kp_inds = 9:11;
req_kp_inds = 9;
kpEnum = makeStringEnum(requiredKeypoints,true,false);

%%
showNN(feats_l2,curImgs,15)
%%
feats_l2 = feats_l2_bkp;

% feats_l2 = normalize_vec(feats_l2,1,1);
feats_l2 = rootsift(feats_l2);
% feats_l2 = vl_homkermap(feats_l2,2);

%%
 %[feats_l2,mu,stds] = normalizeData(feats_l2);
 
 
 feats_l2 = normalizeData2(feats_l2);

%%
% predictors0 = train_predictors(feats_l2,sel_train,all_kps(:,req_kp_inds,:),requiredKeypoints,.0001);
predictors0 = train_predictors(feats_l2,sel_train,all_kps(:,req_kp_inds,:),requiredKeypoints,.001);
preds_xy = apply_predictors(predictors0,feats_l2,sel_test);

% predictors0 = train_predictors_forest(feats_l2,sel_train,all_kps(:,req_kp_inds,:),requiredKeypoints,.0001);
% preds_xy = apply_predictors_forest(predictors0,feats_l2,sel_test);

gt_ = squeeze(all_kps(sel_test,9,:));
p = squeeze(preds_xy);
%
figure(1),hist(vec_norms(p-gt_,2),100)
%%
preds_ = p;
[r,ir] = sort(vec_norms(preds_-gt_,2),'descend');
% 
cur_set = sel_test;
%it = 1:length(cur_set);
figure(2);
for it = 1:50:length(cur_set)    
         t = ir(it);
         if (isnan(gt_(t,1))),continue,end
%     t = ir(it);
    I=curImgs{cur_set(t)};
    cla; imagesc2(I);
    plotPolygons(gt_(t,:),'ro','LineWidth',3);
    for z = 1:length(predictors0)
        %plotPolygons(squeeze(preds_xy(t,z,:))','g+','LineWidth',2);
        plotPolygons(squeeze(preds_(t,:)),'g+','LineWidth',2);
%         squeeze(preds_xy(t,z,:))'
    end    
    dpc
%     dpc(.1)
end
%%

figure,hist(vec_norms(preds_-gt_,2),100)

 [no,xo] = hist(vec_norms(p-gt_,2),100);
 [no2] = hist(vec_norms(preds_-gt_,2),xo);
 
 stem(xo,no,'b'); hold on;
 stem(xo,no2,'r');
 
 plot(cumsum(no)/sum(no));hold on
 plot(cumsum(no2)/sum(no),'r');
 grid minor
 
figure,hist([vec_norms(p-gt_,2),vec_norms(preds_-gt_,2)],100)
% figure,hist(,100)

%%
lambda = .01;
trainOpts = sprintf('-s 13 -B 1 -e %f',lambda);
yaws = [poses.yaw];
yaw_model = train(yaws(sel_train)',sparse(double(feats_l2(:,sel_train))), trainOpts,'col');
predicted_yaw = predict2(feats_l2(:,sel_test),yaw_model.w);
% figure,plot(predicted_yaw,yaws(sel_test),'r+');
std(predicted_yaw-yaws(sel_test)')
figure,hist(predicted_yaw-yaws(sel_test)',50)
% apply predictions, jitter a bit and train again.
%%
feats_l2 = single(feats_l2);
for t = 1:length(predictors0)
    predictors0(t).wx = single(predictors0(t).wx);
    predictors0(t).wy = single(predictors0(t).wy);
end

%boxes = zeros(size(preds_xy,1),4);
preds_xy = apply_predictors(predictors0,feats_l2,sel_train);
boxes = squeeze(mean(preds_xy,2));
all_kps_new = all_kps(:,req_kp_inds,:);
boxes_ = zeros(size(boxes,1),1,size(boxes,2));
boxes_(:) = boxes;
all_kps_new(sel_train,:,:) = bsxfun(@minus,all_kps_new(sel_train,:,:),boxes_);
boxes = inflatebbox(boxes,60,'both',true);
boxes = round(boxes);
I1 = multiCrop2(curImgs(sel_train),boxes);

for t = 1:length(sel_train)
    clf; imagesc2(curImgs{sel_train(t)});
    plotBoxes(boxes(t,:));
    dpc;
end

feats_l2_1 = featureExtractor.extractFeaturesMulti(I1,false);
%feats_l2_1 = normalize_vec(feats_l2_1);
feats_l2_1 = rootsift(feats_l2_1);
%normalize_vec(feats_l2_1);
predictors1 = train_predictors(feats_l2_1,1:size(feats_l2_1,2),all_kps_new(sel_train,:,:),requiredKeypoints,.001);

%%
wSize = 60;
preds_ = {};

for u = 1:length(sel_test)
    if (mod(u,20)==0)
        u
    end
    I = curImgs{sel_test(u)};
    curPred = detect_keypoints(I,predictors0,predictors1,featureExtractor,wSize);
    preds_{u} = curPred;
end

preds_ = cat(1,preds_{:});

%%

% preds_ = p;
[r,ir] = sort(vec_norms(preds_-gt_,2),'descend');
% 
cur_set = sel_test;
%it = 1:length(cur_set);

for it = 1:50:length(cur_set)    
         t = ir(it);
         if (isnan(gt_(t,1))),continue,end
%     t = ir(it);
    I=curImgs{cur_set(t)};
    cla; imagesc2(I);
    plotPolygons(gt_(t,:),'ro','LineWidth',3);
    for z = 1:length(predictors0)
        %plotPolygons(squeeze(preds_xy(t,z,:))','g+','LineWidth',2);
        plotPolygons(squeeze(preds_(t,:)),'g+','LineWidth',2);
%         squeeze(preds_xy(t,z,:))'
    end    
    dpc
%     dpc(.1)
end
%%


%% 
 % predict the bounding box of the face...
 
 train_imgs = {};
 train_boxes = {};
 for t = 1:length(fra_db)
     imgData = fra_db(u);
     faceBox_gt = imgData.faceBox;
     
     [a,b,c] = BoxSize(faceBox_gt);
     center_xy = boxCenters(faceBox_gt);
     
     
     faceBox_raw = imgData.faceBox_raw;
     [I_sub_train,faceBox,mouthBox] = getSubImage2(conf,imgData,false);
 end
    
 
%%
for u = 1:length(fra_db)
    imgData = fra_db(u);
     [I_sub,faceBox,mouthBox] = getSubImage2(conf,imgData,true);
     faceBox = inflatebbox(faceBox,1.5,'both',false)
     I = getImage(conf,imgData);
     faceBox = round(faceBox);
     I = cropper(I,faceBox);
     I = imResample(I,[128 128],'bilinear');
     predictors1 = [];
     %detect_keypoints(I,predictors0,predictors1,featureExtractor,wSize,false,true)
     detect_keypoints(I,predictors0,[],featureExtractor,wSize,false,true)
     dpc
%      globalFeats = getImageStackHOG(I);
%      %globalFeats = featureExtractor.extractFeaturesMulti(I,false);
%      X = normalize_vec(globalFeats,1);
%      cur_preds0 = squeeze(apply_predictors(predictors0,X,1))';
%      clf; imagesc2(I); plotPolygons(cur_preds0,'r+','LineWidth',3);
%      dpc

end

% \th(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta14'))

%L = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta14/pascal-fcn32s-dag.mat')

L = load('~/storage/matconv_data/pascal-fcn8s-tvg-dag.mat');
net = dagnn.DagNN.loadobj(L);
vl_setupnn
I = getImage(conf,imgData);

net.mode = 'test';
net.move('gpu');

I = gpuArray(I);
net.eval({'data',single(I)});
[r,ir] = max(net.vars(end).value,[],3);
% x2(net.vars(end).value(:,:,16)>.9)
%x2(exp(net.vars(end).value(:,:,16))); colorbar
x2((net.vars(end).value(:,:,2))); colorbar

v = (net.vars(end).value);
% v = bsxfun(@rdivide,v,sum(v,3));
x2(v(:,:,16)); colorbar


x2(I)

net.meta.classes.name(16)
%net = dagnn.DagNN.fromSimpleNN(net, 'canonicalNames', true) ;
