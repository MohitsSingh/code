%% demo_withFace2
%
initpath;
config;
% precompute the cluster responses for the entire training set.
%
conf.suffix = 'train_dt_noperson';
baseSuffix = 'train_noperson_top_nosv';
conf.suffix = baseSuffix;

conf.VOCopts = VOCopts;
% dataset of images with ground-truth annotations
% start with a one-class case.

[train_ids,train_labels] = getImageSet(conf,'train',1,0);
[test_ids,test_labels] = getImageSet(conf,'test');
conf.detetion.params.detect_max_windows_per_exemplar = 1;

conf.detection.params.max_models_before_block_method = 0;
conf.max_image_size = 100;
conf.clustering.num_hard_mining_iters = 10;
k = 10;
curSuffix = num2str(k);
c_trained = train_patch_classifier(conf,[],getNonPersonIds(VOCopts),'suffix',['face_' curSuffix],'override',false);
conf.max_image_size = 256;

%%
%%
f = find(train_labels);
conf.detection.params.detect_min_scale = .5;
conf.max_image_size = 256;

[qq1,q1,aps] = applyToSet(conf,c_trained,train_ids,[],'c_check','override',false,'uniqueImages',true,...
    'nDetsPerCluster',10,'disp_model',true,'visualizeClusters',false);

[qq1_test,q1_test,aps_test] = applyToSet(conf,c_trained,test_ids,[],'c_check_test','override',false,'uniqueImages',true,...
    'nDetsPerCluster',10,'disp_model',true);

k  =4;
train_true_labels = train_labels(qq1(k).cluster_locs(:,11));

load m_train.mat;
load m_test;

% get landmarks...
% landmarks_train = detect_landmarks(conf,m_train,2);
% landmarks = detect_landmarks(conf,train_ids);
% save landmarks_train landmarks
% load landmarks;
% sz = [20 40];
% locs = allBoxes;
% locs(:,11) = 1:4000;
% lipImages = visualizeLocs2_new(conf,train_ids,locs,'add_border',false);

% lipImages = multiCrop(train_ids,round(inflatebbox(allBoxes,3)),sz);

% save landmarks_train.mat landmarks; % 30/1/2013
%%
% train a classifier to tell the difference between drinking / non
% drinking. 

[faceLandmarks,allBoxes] = landmarks2struct(landmarks_train);

save allLipsBoxes allBoxes_complete

sz = [20 40];
[lipImages_train,faceScores_train] = getLipImages(m_train,landmarks_train,sz);
[lipImages_test,faceScores_test] = getLipImages(m_test,landmarks_test,sz);

lipImages = multiCrop(m1_train,round(inflatebbox(allBoxes_complete,4)),sz);

lipDet = qq1_lip(1);
[~,c2,] = intersect(lipDet.cluster_locs(:,11),1:1000);
bb2 = lipDet.cluster_locs(c2,:);

lipImages_q = visualizeLocs2_new(conf2,m1_train,bb2);
imshow(multiImage(lipImages(1:100),false))

f = find(train_true_labels);
f_ = find(~train_true_labels);
imshow(multiImage(lipImages(1:100),false))

imwrite(multiImage(lipImages(f(1:50)),false),'drinking_lips_c.tif');
imwrite(multiImage(lipImages(f_(1:50)),false),'not_drinking_lips_c.tif');

X = imageSetFeatures2(conf,lipImages);
% X = imageSetFeatures2(conf,lipImages_q);
X = cat(2,X{:});

X1 = vl_hog(im2single(lipImages{1}),conf.features.vlfeat.cellsize,'NumOrientations',9);

train_true_labels = train_labels(qq1(k).cluster_locs(1:1000,11));

% ttt = train_true_labels & (qq1(4).cluster_locs(1:1000,12)>.15);
ttt = train_true_labels;

[ws1,b1,sv,coeff] = train_classifier(X(:,ttt),X(:,~ttt),.01,10);

hogPic = jettify(HOGpicture(reshape(ws1(1:end),size(X1)),20));
figure,imshow(hogPic);

% apply to all images...
scores = ws'*X_test-b+0*s(t==2);
figure,plot(scores);
[r,ir] = sort(scores,'descend');
n = 100;
lipImagesTest = lipImages(t==2);
hh = multiImage(lipImagesTest(ir(1:n)),false);

figure,imshow(hh)
imwrite(hh,'drinking_candidates.tif')

save drikingMouthDetector.mat ws1 b1

%% check on test images....
load m_test
[~,ic,ib] = intersect(qq1_test(4).cluster_locs(:,11),1:length(m_test));
m_test_t = m_test(ic);
m_test_true = m_test_t(test_labels);
imshow(multiImage(m_test_true(1:50)));
toScale =1;
landmarks_test = detect_landmarks(conf,m_test_t,2);
load landmarks_train.mat;
load  m_test_t_landmarks.mat
% save m_test_t_landmarks.mat landmarks_test
[lm_s,testBoxes] = landmarks2struct(landmarks_test);
for k = 1:length(lm_s);
    imshow(m_test_t{k});
    hold on;
    plotBoxes2(lm_s(k).lipBox([2 1 4 3])/2);
    pause;
end

sz = [20 40];
lipImagesTest = multiCrop(m_test_t,round(inflatebbox(testBoxes/2,4)),sz);

figure,imshow(m_test_t{1500})
figure,imshow(getImage(conf,test_ids{1500}))

X_test = imageSetFeatures2(conf,lipImagesTest);
X_test = cat(2,X_test{:});

%%
scores = ws1'*X_test-b1 +2*double(qq1_test(4).cluster_locs(ic,12)'>.1);
% 0*.1*double([lm_s.s]>-.8);
%
[r,ir] = sort(scores,'descend');
n = 100;

n = 100;
toPaint = lipImagesTest(ir(1:n));
scr
for kk = 1:n
    if test_labels(ib(ir(kk)))
        toPaint{kk} = imresize(addBorder(toPaint{kk},2,[0 255 0]),1);
    end
end

hh = multiImage(toPaint,false);

figure,imshow(hh)

scores_ = zeros(size(scores));
scores_(ib) = scores;

[prec,rec,aps] = calc_aps2(scores_(:),test_labels);
aps

%%
% now train a classifier to tell the difference between interacting mouth /
% non interacting. 
% interacting categories are: 
% drinking 9
% brushing teeth 3
% blowing bubbles?  2
% smoking 30

t_all = train_labels;
conf.class_subset = 2;
[~,t_blowing] = getImageSet(conf,'train');
conf.class_subset = 3;
[~,t_brusing] = getImageSet(conf,'train');
conf.class_subset = 30;
[~,t_smoking] = getImageSet(conf,'train');
% revert to drinking :-)
conf.class_subset = 9;

t_all= t_all | t_blowing | t_smoking | t_brusing;

% get the test labels as well...
t_all_test = test_labels;
conf.class_subset = 2;
[~,t_blowing_test ] = getImageSet(conf,'test');
conf.class_subset = 3;
[~,t_brusing_test ] = getImageSet(conf,'test');
conf.class_subset = 30;
[~,t_smoking_test ] = getImageSet(conf,'test');
% revert to drinking :-)
conf.class_subset = 9;

t_all_test = t_all_test | t_blowing_test | t_smoking_test | t_brusing_test;

t_all_1 = t_all(qq1(4).cluster_locs(1:1000,11));

imwrite(multiImage(m1_train(t_all_1)),'multi.tif');
imwrite(multiImage(lipImages(t_all_1),false),'multi_lips.tif');

X_d = imageSetFeatures2(conf,lipImages);
X_d = cat(2,X_d{:});
[ws_d,b_d] = train_classifier(X(:,t_all_1),X(:,~t_all_1),.01,1);

hogPic = jettify(HOGpicture(reshape(ws_d(1:end),size(X1)),20));
figure,imshow(hogPic);

imwrite(hogPic,'hog_interaction.tif')

%%... and test on test set. 
%%
scores_i = ws_d'*X_test-b_d +4*double(qq1_test(4).cluster_locs(ic,12)'>0);
% 0*.1*double([lm_s.s]>-.8);
%
[r,ir] = sort(scores_i,'descend');

n = 100;
toPaint = lipImagesTest(ir(1:n));

for kk = 1:n
    if t_all_test(ib(ir(kk)))
        toPaint{kk} = imresize(addBorder(toPaint{kk},2,[0 255 0]),1);
    end
end

hh = multiImage(toPaint,false);

figure,imshow(hh)

scores_ = zeros(size(scores_i));
scores_(ib) = scores_i;

[prec,rec,aps] = calc_aps2(scores_(:),t_all_test);
aps
figure,plot(rec,prec)

imwrite(hh,'interacting_mouths.tif')

%% now add to score the drinking score....
scores_drink = ws1'*X_test-b1+1*double(qq1_test(4).cluster_locs(ic,12)'>.1);
scores_drink = scores_drink+1*5*scores_i;
% 0*.1*double([lm_s.s]>-.8);
%
[r,ir] = sort(scores_drink,'descend');
n = 100;

n = 100;
toPaint = lipImagesTest(ir(1:n));

for kk = 1:n
    if test_labels(ib(ir(kk)))
        toPaint{kk} = imresize(addBorder(toPaint{kk},2,[0 255 0]),1);
    end
end

hh = multiImage(toPaint,false);

figure,imshow(hh)

scores_ = zeros(size(scores_drink));
scores_(ib) = scores_drink;

[prec,rec,aps] = calc_aps2(scores_(:),test_labels);
aps

imwrite(hh,'drinking_from_interaction.jpg')

allTrainLabels = getAllLabels(conf,'train');
allTestLabels = getAllLabels(conf,'test');

[q,iq] = sort(scores_,'descend');
iq(1:100)
figure,hist(allTestLabels(iq(1:100)),1:40)

ss = scores_-min(scores_);
ss = ss/sum(ss);
figure,hist(ss)

%%

figure,plot([faceLandmarks.s])

[~,ir] = sort([faceLandmarks.s],'descend');
r = multiImage(m1_train,false);
imwrite(r,'fff.tif');

a = multiImage(lipImages,false);
imwrite(imresize(a,2),'lll.tif');

a1 = multiImage(lipImages(train_true_labels),false);
imwrite(imresize(a1,2),'lll_true.tif');

%%
for q = 1:length(m1_train)
%     if (~train_true_labels(q))
%         continue;
%     end
    im = m1_train{q};
    im = imresize(im,1);
    %bs(1).xy = bs(1).xy/2;
    clf;
    imshow(im);
    bs = landmarks{q};
    if (isempty(bs))
        disp(['no face found for example ' num2str(q)]);
        continue
    end    
    
    bs(1).xy = bs(1).xy /2;
    lipCoords = bs(1).xy(33:51,:);
    lipPoints = boxCenters(lipCoords);
    showboxes(im, bs(1),posemap);
    hold on;
    plot(lipPoints(:,1),lipPoints(:,2),'m*');
    pause;
end

%%
% try  a nerest neighbor classifier...
R_pos = l2(X_test',X(:,ttt)');
R_neg = l2(X_test',X(:,~ttt)');

% sort according  to distance to class 1
[m_pos,im_pos] = min(R_pos,[],2);
[m_neg,im_neg] = min(R_neg,[],2);
sig_ = 10;
rrr = find(test_labels(qq1_test(4).cluster_locs(ic,11)));
nn_scores = -m_pos;
nn_scores_neg = -m_neg;
% exp(-m_pos/sig_);
[r,ir] = sort(nn_scores,'descend');


% ir = rrr;

% exp(-m_pos/sig_)./(exp(-m_pos/sig_)+exp(-m_neg/sig_));
% % lipsPos = lipImages(ttt);
% % figure,imshow(multiImage(lipsPos(im_pos(ir(1:100))),false)); title('pos neighbors');
% % % [r_neg,ir_neg] = sort(nn_scores,'descend');
% % lipsNeg = lipImages(~ttt);
% % figure,imshow(multiImage(lipsNeg(im_neg(ir(1:100))),false));title('neg neighbors');
% % figure,imshow(multiImage(lipImagesTest(ir(1:100)),false));

%%
nn_scores = exp(-m_pos/sig_)./(exp(-m_pos/sig_)+exp(-m_neg/sig_));
nn_scores = nn_scores +2*double(qq1_test(4).cluster_locs(ic,12)'>.05)';
[~,ir] = sort(nn_scores,'descend');
%
figure,imshow(multiImage(lipImagesTest(ir(1:100)),false));

scores_ = zeros(size(test_labels));
scores_(qq1_test(4).cluster_locs(ic,11)) = nn_scores;
% scores_(ib) = scores_(ib)+.04*scores2(:);
%scores_(qq1_test(4).cluster_locs(:,11)) =  +scores2;
[prec,rec,aps] = calc_aps2(scores_,test_labels);
% figure,plot(rec,prec)
aps


%%
%%  train hog for entire face...
X = imageSetFeatures2(conf,m1_train);
X = cat(2,X{:});
[ws2,b2] = train_classifier(X(:,train_true_labels),X(:,~train_true_labels),.01,1);
%%
X2= imageSetFeatures2(conf,m_test);
X2 = cat(2,X2{:});
%%
scores = ws2'*X2-b2+ 4*double(qq1_test(4).cluster_locs(:,12)'>0.1);
scores2 = ws1'*X_test;

% scores = ws2'*X2-b2+ .4*qq1_test(4).cluster_locs(:,12)';

[r,ir] = sort(scores,'descend');
n = 64;

[~,ic,ib] = intersect(qq1_test(4).cluster_locs(:,11),1:length(m_test));
toPaint = m_test(ir(1:n));
tt = test_labels(qq1_test(4).cluster_locs(:,11));
for kk = 1:n
    if tt(ir(kk))
        toPaint{kk} = imresize(addBorder(toPaint{kk},3,[0 255 0]),1);
    else
        toPaint{kk} = imresize(addBorder(toPaint{kk},3,[255 0 0]),1);
    end
end

figure,imshow(multiImage(toPaint))

scores_ = zeros(size(test_labels));
scores_(qq1_test(4).cluster_locs(:,11)) = scores;
% scores_(ib) = scores_(ib)+.04*scores2(:);
%scores_(qq1_test(4).cluster_locs(:,11)) =  +scores2;
[prec,rec,aps] = calc_aps2(scores_,test_labels);
% figure,plot(rec,prec)
aps
%%

