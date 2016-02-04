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


f = find(train_labels);
conf.detection.params.detect_min_scale = .5;
conf.max_image_size = 256;

[qq1,q1,aps] = applyToSet(conf,c_trained,train_ids,[],'c_check','override',false,'uniqueImages',true,...
    'nDetsPerCluster',10,'disp_model',true,'visualizeClusters',false);

[qq1_test,q1_test,aps_test] = applyToSet(conf,c_trained,test_ids,[],'c_check_test','override',false,'uniqueImages',true,...
    'nDetsPerCluster',10,'disp_model',true);

k  =4;

load m_test
load m_train.mat
[~,ic,~] = intersect(qq1_test(4).cluster_locs(:,11),1:length(m_test));
m_test_t = m_test(ic);
m_test_true = m_test_t(test_labels);
imshow(multiImage(m_test_true(1:50)));
toScale =1;
% landmarks_test = detect_landmarks(conf,m_test_t,2);
load landmarks_train.mat;
load  m_test_t_landmarks.mat

%%
qChoice = 4;
conf.features.vlfeat.cellsize =8;
train_true_labels = train_labels(qq1(qChoice).cluster_locs(:,11));% for m_train.

faceDScores_train = qq1(qChoice).cluster_locs(:,12);
[~,ic,ib] = intersect(qq1_test(qChoice).cluster_locs(:,11),1:length(m_test));
m_test_t = m_test(ic);
test_true_labels = test_labels(qq1_test(qChoice).cluster_locs(:,11));
test_true_labels = test_true_labels(ic);
faceDScores_test = qq1_test(qChoice).cluster_locs(ic,12);

sz = [20 40];
inflateFactor = 3; % enlarge area around lips by this much.
%load landmarks_train
%load  m_test_t_landmarks.mat
[lipImages_train,faceScores_train] = getLipImages(m_train,landmarks_train,sz,inflateFactor,train_true_labels);
[lipImages_test,faceScores_test] = getLipImages(m_test_t,landmarks_test,sz,inflateFactor);
flat = true;


conf.features.vlfeat.cellsize = 8;
hog_train = imageSetFeatures2(conf,lipImages_train,flat);
hog_test = imageSetFeatures2(conf,lipImages_test,flat);
% the set of all actions in which mouth is "active":
% blowing bubbles, drinking, smoking and brushing teeth.
%
conf.class_subset = [2 3 30 9];
[~,labels_active_train] = getImageSet(conf,'train');
labels_active_train = labels_active_train(qq1(qChoice).cluster_locs(:,11));
[~,labels_active_test] = getImageSet(conf,'test');
labels_active_test = labels_active_test(qq1_test(qChoice).cluster_locs(:,11));
labels_active_test = labels_active_test(ic);



%% Experiment 2: use both face scores.

T = -.8;
T_d = -.8;
%
feat_train = [hog_train;...
    row(faceScores_train)>T;...
    row(faceDScores_train)>T_d];
feat_test = [hog_test;...
    row(faceScores_test)>T;...
    row(faceDScores_test)>T_d];

sel_train = faceScores_train(:) > T & faceDScores_train > T_d;
sel_test = faceScores_test(:) > T & faceDScores_test > T_d;

t_train = train_true_labels(sel_train);
t_test= test_true_labels(sel_test);

lipImages_train_s = lipImages_train(sel_train);
DPM_path = '/home/amirro/code/3rdparty/voc-release4.01/';
addpath(genpath(DPM_path));

lipImages_train_s_2 = multiCrop(lipImages_train_s,[],3*[40 80]);

lipImages_train_s_2_ = lipImages_train_s_2(1:10:end);
t_train_ = t_train(1:10:end);
model_lips = trainDPM(conf,lipImages_train_s_2_(t_train_),lipImages_train_s_2_(~t_train_),'drinking_lips');

visualizemodel(model_lips)


lipImages_test_s = lipImages_test(sel_test);
lipImages_test_s_2 = multiCrop(lipImages_test_s,[],3*[40 80]);
tt = {}
rr = lipImages_test_s_2;
parfor iii = 1:length(rr)
    im = rr{iii};
    iii
    [dets, boxes] = imgdetect(im, model_lips, -1.5);
    top = nms(dets, 0.5);
    tt{iii} = dets(top(1),end);
%     boxes(top(1),:)
%     showboxes(im, reduceboxes(model_lips, boxes(top(1),:)));
%     title(num2str(dets(top(1),end)));
%     pause;
end
tt_ = cat(1,tt{:});
[q,iq] = sort(tt_,'ascend');
addpath('utils')
figure,imshow(multiImage(lipImages_test_s_2(iq(1:300)),false));

% III = multiCrop(lipImages_train_s(t_train),[1 1 40 20],[40 80]);


% check the results...
test_scores = ws_2'*feat_test_t-b_2;
[r,ir] = sort(test_scores,'descend');
mm_test = m_test_t(sel_test);
mm_test_toshow = lipImages_test(sel_test);
mm_test_toshow = paintRule(mm_test,t_test);
% for k = 1:length(mm_test)
%     if (t_test(k))
%         mm_test_toshow{k} = addBorder(mm_test_toshow{k},3,[0 255 0]);
%     end
% end

% figure,imshow(
figure,imshow(multiImage(mm_test_toshow(ir(1:200)),false));
imwrite(multiImage(mm_test_toshow(ir(1:200)),false),'detections_painted.jpg');
scores_ = -inf*ones(size(test_true_labels));
scores_(sel_test) = test_scores;
%scores_(ib(sel_test)) = test_scores;
p = randperm(length(test_true_labels)); % note that the precision here
% is measured w.r.t the actual number of true instances in the remaining
% selection
[prec,rec,aps] = calc_aps2(scores_(p),test_true_labels(p),sum(t_test));
figure,plot(rec,prec)
aps

%% %% Experiment 3: learn "active mouths" instead of drinking.
append_faceScore =0; % add the face (landmarks) score as an additional feature. includes weight as well.
append_faceDScore =0; % add the face (detection) score as an additional feature. includes weight as well.

feat_train = hog_train;
feat_test = hog_test;

T = -.8;
T_d = -.8;

sel_train = faceScores_train(:) > T & faceDScores_train > T_d;
sel_test = faceScores_test(:) > T & faceDScores_test > T_d;

feat_train_t = feat_train(:,sel_train);
feat_test_t = feat_test(:,sel_test);
t_train = labels_active_train(sel_train);
t_test = labels_active_test(sel_test);

[ws_2,b_2] = train_classifier(feat_train_t(:,t_train),feat_train_t(:,~t_train),.01,10);

x1 = vl_hog(im2single(lipImages_train{1}),conf.features.vlfeat.cellsize,'NumOrientations',9);
sz_ = size(x1);
figure,imshow(jettify(HOGpicture(reshape(ws_2(1:end),sz_),20)));

% check the results...
test_scores = ws_2'*feat_test_t-b_2;
[r,ir] = sort(test_scores,'descend');
mm_test = m_test_t(sel_test);
mm_test_lips = lipImages_test(sel_test);

% mm_test = lipImages_test(sel_test);
mm_test_toshow = paintRule(mm_test,t_test);
% m_ = (multiImage(mm_test_toshow(ir(1:500)),false));
% imwrite(m_,'active_mouths.jpg');
% figure,imshow(multiImage(mm_test_toshow(ir(1:50)),false));
mm_test_lips_show = paintRule(mm_test_lips,t_test);
figure,imshow(multiImage(mm_test_lips_show(ir(1:50)),false));

scores_ = -inf*ones(size(test_true_labels));
scores_(sel_test) = test_scores;
%scores_(ib(sel_test)) = test_scores;
p = randperm(length(labels_active_test));
[prec,rec,aps] = calc_aps2(scores_(p),labels_active_test(p));
figure,plot(rec,prec)
hold on;
plot(rec,sort(scores_,'descend'),'r');
aps

%% Experiement 4: based on previous experiments, now check the local appearance
%% of the lips area and find cups, straws or bottles.
% This is done using multiple-instance learning.
% % segment the lip areas.
lipImages_train_s  = lipImages_train(sel_train);

localLipScore_train = -ones(size(sel_train));
find_sel_train = find(sel_train);
patchAttributes = {};
parfor k = 1:length(lipImages_train_s)
    %     if (t_train(find_sel_train(k)))
    k
    %         k = 73
    % k= 33
    patchAttributes{k} = getPatchAttributes(lipImages_train_s{k});
    %     end
end

imshow(multiImage(lipImages_train_s(t_train),false))

%%
close all;
ff = find(t_train);
fff = find(sel_train);
for k = 1:length(lipImages_train_s(t_train))
    %     if (t_train(find_sel_train(k)))
    k
    getPatchAttributes(lipImages_train_s{ff(k)});
    %         getPatchAttributes(m_train{fff(ff(k))});
end
%%
%%
[d,pixx,attr,contrasts] = verifyStraws(lipImages_train_s, patchAttributes)
%%
d_total = aggregateResults(attr,[1 1 1 0]);

% d_total = d+1*(contrasts);%0*double(contrasts>0);
% d_total(isinf(contrasts)) = -inf;
[r,ir] = sort(d_total,'descend');

figure,imshow(multiImage(jettify(pixx(ir(1:150))),false));

[mm,~,xx,yy] = multiImage(lipImages_train_s(ir(1:150)),false);
figure,imshow(mm);
for kk = 1:length(lipImages_train_s(ir(1:150)))
    text(xx(kk),5+yy(kk),num2str(kk),'color','y','FontSize',10);
end

[prec,rec,aps] = calc_aps2(d_total(:),t_train);
% figure,imshow(multiImage(lipImages_test_s(t_test),false))
figure,plot(rec,prec)
aps

%%
f_true = find(t_train);
figure,imshow(multiImage(lipImages_train_s(f_true),false));
figure,imshow(multiImage(jettify(pixx(f_true)),false));

% f_true(2)
%  getPatchAttributes(lipImages_train_s{261});

%%

% 36-->626
% 29 --> 504
getPatchAttributes(lipImages_train_s{504});
[d,pixx,attr,contrasts] = verifyStraws(lipImages_train_s(t_train), patchAttributes(t_train));
figure,imshow(multiImage(lipImages_train_s(t_train),false));
figure,imshow(multiImage(jettify(pixx),false));

%%
III = multiCrop(lipImages_train_s(t_train),[1 1 40 20],[40 80]);
figure,imshow(multiImage(III))

m_0 = multiImage(III);
figure,imshow(rgb2gray(m_0));
H = rgb2hsv(m_0);
figure,imagesc(H(:,:,3))
figure,imagesc(im2double(rgb2gray(m_0)))
% figure,imagesc(sum(im2double(m_0).^2,3).^.5 > .8)
figure,imagesc(m_0>128)

m_1 = im2double(III{29});

III_test = multiCrop(lipImages_test_s(t_test),[1 1 40 20],[60 107]);
figure,imshow(multiImage(III_test,find(t_test)))

% figure,plot(r(~isinf(r)))
% figure,plot(sort([contrasts{~isinf(d)}]))

getPatchAttributes(lipImages_train_s{ir(83)});
%%
patchAttributesTest = {};
lipImages_test_s =lipImages_test(sel_test);
parfor k = 1:length(lipImages_test_s)
    k
    %         k = 340
    %     k=ir(149)
    patchAttributesTest{k} = getPatchAttributes(lipImages_test_s{k});
end
%%

%%
[d_test,pixx_test,attr_test,contrasts_test] = verifyStraws(lipImages_test_s, patchAttributesTest);

aaa = cellfun(@(x) length(x),attr_test);

% verifyStraws(lipImages_test_s, patchAttributesTest);
% verifyStraws(lipImages_test_s(380), patchAttributesTest(380));

%  getPatchAttributes(lipImages_test_s{380});

%%
d_test_total = aggregateResults(attr_test,[1 1 1 0]);
d_test_total = d_test_total +0*1.5*test_scores(1:length(d_test_total))';

% d_test_total = d_test+1*(contrasts_test)+test_scores*.1;
% d_test_total(isinf(contrasts_test)) = -inf;

[r,ir] = sort(d_test_total,'descend');
% m_test_t_s = m_test_t(sel_test);

lipImages_test_s_p = paintRule(lipImages_test_s,t_test,[],[],1);

% ir = [328 330 332 336 339 340 345 346 349 351 355 357 359 366 370 373 375 380];
% ir(end+1:end+150) = 1;
ir = ir(1:min(100,length(ir)));

figure,imshow(multiImage(lipImages_test_s_p(ir),false));
% figure,imshow(multiImage(m_test_t_s(ir(1:50)),false));
figure,imshow(multiImage(jettify(pixx_test(ir)),false));

% figure,plot(r(~isinf(r)))

% scores_test(sel_test) = d_test;
[prec,rec,aps] = calc_aps2(d_test_total(:),t_test);
% figure,imshow(multiImage(lipImages_test_s(t_test),false))
figure,plot(rec,prec)
aps


%%
mm_0 = multiImage(lipImages_test_s_p(ir),false);
mm_1 = multiImage(jettify(pixx_test(ir)),false);

imwrite(mm_0,'straw_mircs_test.tif');
imwrite(mm_1,'straw_mircs_test_segs.tif');

lipImages_test_s_p_masked = lipImages_test_s_p;
for k = 1:length(pixx_test)
    k
    if (~isempty(pixx_test{k}))
        lipImages_test_s_p_masked{k}=  repmat(im2double(pixx_test{k}>0),[1,1,3]).*...
            im2double(lipImages_test_s_p{k});
    end
end

mm_1 = multiImage(lipImages_test_s_p_masked(ir),false);


imwrite(mm_1,'straw_mircs_test_segs_masked.tif');

%% Experiement 5 based on previous experiments, now check the local appearance
%% of the lips area and find cups
patchAttributes_c = {};
parfor k = 1:length(lipImages_train_s)
    %     if (t_train(find_sel_train(k)))
    k
    %         k = 73
    % k= 33
    patchAttributes_c{k} = getPatchAttributes_cup(lipImages_train_s{k});
    %     end
end
%%
% r = getPatchAttributes_cup(lipImages_train_s{f_true(5)});
%  verifyCups(lipImages_train_s(f_true(5)), {r})
[d,pixx,attr,contrasts] = verifyCups(lipImages_train_s, patchAttributes_c)

%%
d_total = aggregateResults(attr,[10 1 -1 10]);
test_scores
% d_total = d+1*(contrasts);%0*double(contrasts>0);
% d_total(isinf(contrasts)) = -inf;

[r,ir_] = sort(d_total,'descend');

ir = f_true;
ir(end+1:150) = ir_(length(ir)+1:150);

figure,imshow(multiImage(jettify(pixx(ir(1:150))),false));

[mm,~,xx,yy] = multiImage(lipImages_train_s(ir(1:150)),false);
figure,imshow(mm);
for kk = 1:length(lipImages_train_s(ir(1:150)))
    text(xx(kk),5+yy(kk),num2str(kk),'color','y','FontSize',10);
end

[prec,rec,aps] = calc_aps2(d_total(:),t_train);
% figure,imshow(multiImage(lipImages_test_s(t_test),false))
% figure,plot(rec,prec)
aps
%%
getPatchAttributes_cup(lipImages_train_s{ir(5)})


%% try again using Bag-of-Words
sz = size(lipImages_test_s{1});
aaa = multiCrop(lipImages_train_s,[],2*sz(1:2));
bbb = multiCrop(lipImages_test_s,[],2*sz(1:2));

% figure,imshow(multiImage(aaa))


dict = learnBowDictionary(conf,aaa,true);
model.numSpatialX = [2];
model.numSpatialY = [2];
model.kdtree = vl_kdtreebuild(dict) ;
model.quantizer = 'kdtree';
model.vocab = dict;
model.w = [] ;
model.b = [] ;
model.phowOpts = {};

hists_train = getHists(conf,model,aaa);
hists_test = getHists(conf,model,bbb);

y_train = 2 * (t_train(:)==1) - 1;
y_test = 2 * (t_test(:)==1) - 1;
%%

%%
psix_train = vl_homkermap(hists_train, 1, 'kchi2', 'gamma', .5) ;
psix_test = vl_homkermap(hists_test, 1, 'kchi2', 'gamma', .5) ;
% svmModel= svmtrain(y_train(:), double(psix_train'),'-t 2');
% psix_test = vl_homkermap(hists_test, 1, 'kchi2', 'gamma', .5)';
%    dd = psix_test*w;

%%
% [w b info] = vl_pegasos(psix_train,int8(y_train),.00001);
svmModel = svmtrain(y_train, double(psix_train'),'-t 0');
[~, ~, decision_values_train] = svmpredict(zeros(size(psix_train,2),1),double(psix_train'),svmModel);
decision_values_train = decision_values_train*svmModel.Label(1);
[r_s,ir_s] = sort(decision_values_train,'descend');
[mm,~,xx,yy] = multiImage(lipImages_train_s(ir_s(1:150)),false);
figure,imshow(mm)

[~, ~, decision_values_test] = svmpredict(zeros(size(psix_test,2),1),double(psix_test'),svmModel);
%%
close all
total_score = 0*0.001*test_scores(:)-decision_values_test;

[r_s,ir_s] = sort(total_score,'descend');
lipImages_test_s_show = paintRule(lipImages_test_s,t_test);
[mm,~,xx,yy] = multiImage(lipImages_test_s_show(ir_s(1:150)),false);
figure,imshow(mm)

% [prec,rec,aps] = calc_aps2(scores_,test_labels);
[prec,rec,aps] = calc_aps2(total_score,t_test);
% figure,imshow(multiImage(lipImages_test_s(t_test),false))
figure,plot(rec,prec); title(num2str(aps));


%% Experiment 7: break drinking into several classes, and use scanning window!

sz_1 = [20 40]*1.5;
inflateFactor = 4; % enlarge area around lips by this much.
%load landmarks_train
%load  m_test_t_landmarks.mat
[lipImages_train_1] = getLipImages(m_train,landmarks_train,sz_1,inflateFactor,train_true_labels);
[lipImages_test_1] = getLipImages(m_test_t,landmarks_test,sz_1,inflateFactor);
flat = true;

append_faceScore =0; % add the face (landmarks) score as an additional feature. includes weight as well.
append_faceDScore =0; % add the face (detection) score as an additional feature. includes weight as well.


T = -.8;
T_d = -.8;
%
feat_train = [hog_train];
feat_test = [hog_test];

sel_train = faceScores_train(:) > T & faceDScores_train > T_d;
sel_test = faceScores_test(:) > T & faceDScores_test > T_d;

% sel_train = 1:length(train_true_labels);
% sel_test = 1:length(test_true_labels);

feat_train_t = feat_train(:,sel_train);
feat_test_t = feat_test(:,sel_test);
t_train = train_true_labels(sel_train);
t_test = test_true_labels(sel_test);


sz = size(lipImages_train_s{1});
aaa = multiCrop(lipImages_train_s,[],3*sz(1:2));
% figure,imshow(multiImage(aaa(t_train)))
it_train = find(t_train);

f_train = find(sel_train);
f_test = find(sel_test);
lipImages_train_1(sel_train)

straw_indices = [1 2 4 11 15 20 29 36 43 46];
straw_indices = 2;
t_train_1 = false(size(t_train));
t_train_1(it_train(straw_indices))= true;

[straw_clust] = makeCluster(feat_train_t(:,t_train_1),[]);
conf_s = conf;
conf_s.clustering.num_hard_mining_iters = 10;
conf_s.features.winsize = [3 5];
straw_trained = train_patch_classifier(conf_s,straw_clust,lipImages_train_1(f_train(~t_train)),...
    'suffix','straw_train','override',true);

figure,imshow(showHOG(conf_s,straw_trained));
conf_s.detection.params.detect_add_flip = true;
conf_s.detection.params.detect_min_scale =.7;
res_straw = applyToSet(conf_s,straw_trained,lipImages_test_1(f_test),[],'straw_test_res','toSave',false);
res_straw_test = visualizeLocs2_new(conf_s,lipImages_test_1(f_test),res_straw.cluster_locs,'add_border',false);
figure,imshow(multiImage(res_straw_test(1:100)))


%% Experiment 6: break drinking into several classes.
append_faceScore =0; % add the face (landmarks) score as an additional feature. includes weight as well.
append_faceDScore =0; % add the face (detection) score as an additional feature. includes weight as well.


T = -.8;
T_d = -.8;
%
feat_train = [hog_train];
feat_test = [hog_test];

sel_train = faceScores_train(:) > T & faceDScores_train > T_d;
sel_test = faceScores_test(:) > T & faceDScores_test > T_d;

% sel_train = 1:length(train_true_labels);
% sel_test = 1:length(test_true_labels);

feat_train_t = feat_train(:,sel_train);
feat_test_t = feat_test(:,sel_test);
t_train = train_true_labels(sel_train);
t_test = test_true_labels(sel_test);

sz = size(lipImages_train_s{1});
aaa = multiCrop(lipImages_train_s,[],3*sz(1:2));
% figure,imshow(multiImage(aaa(t_train)))
it_train = find(t_train);

straw_indices = [1 2 4 11 15 20 29 36 43 46];
% straw_indices = [1 2 4 11 16 22 30 53 67 86];
[ws_straw,b_straw] = train_sub_classifier(feat_train_t,t_train,straw_indices);
cup_indices = [5 6 7 9 14 17 18 23 28 31 39 40 41 44 45 47];
% cup_indices = [5 6 7 9 14 17 19 20 26 35 40 46 59 63 69 70 74 79 87];
[ws_cup,b_cup] = train_sub_classifier(feat_train_t,t_train,cup_indices);
other_indices = setdiff(1:length(it_train),[straw_indices cup_indices]);
[ws_other,b_other] = train_sub_classifier(feat_train_t,t_train,other_indices);

x1 = vl_hog(im2single(lipImages_train{1}),conf.features.vlfeat.cellsize,'NumOrientations',9);
sz_ = size(x1);
figure,imshow(jettify(HOGpicture(reshape(ws_straw(1:end),sz_),20)));
figure,imshow(jettify(HOGpicture(reshape(ws_cup(1:end),sz_),20)));
figure,imshow(jettify(HOGpicture(reshape(ws_other(1:end),sz_),20)));


train_scores_straw = ws_straw'*feat_train_t-b_straw;
train_scores_cup = 0*ws_cup'*feat_train_t-b_cup;
train_scores_other = 0*ws_other'*feat_train_t-b_other;
%%
feats = [train_scores_straw;train_scores_cup;train_scores_other;...
    faceScores_train(sel_train);faceDScores_train(sel_train)'];
% [ws_total,b_total] = train_sub_classifier(feats,t_train);

ss = sprintf(['-s 0 -t 0 -c' ...
    ' %f -w1 %.9f -q'], .01, 10);
svm_model= svmtrain(2*(t_train==1)-1, double(feats'),ss);
% [~, ~, decision_values_train] = svmpredict(zeros(size(psix_train,2),1),double(psix_train'),svmModel);

w = -svm_model.SVs'*svm_model.sv_coef;
% svm_weights = full(sum(svm_model.SVs .* ...
%                          repmat(svm_model.sv_coef,1, ...
%                                 size(svm_model.SVs,2)),1));
%
% ws = w(:);
% b = svm_model.rho;
% sv = svm_model.SVs';
% coeff = svm_model.sv_coef;
%
% figure,imagesc(feats)

% check the results...

test_scores_straw = ws_straw'*feat_test_t-b_straw;
test_scores_cup = ws_cup'*feat_test_t-b_cup;
test_scores_other = ws_other'*feat_test_t-b_other;

feats_test = [test_scores_straw;test_scores_cup;test_scores_other;...
    faceScores_test(sel_test);faceDScores_test(sel_test)'];
[~, ~, test_scores] = svmpredict(zeros(size(feats_test,2),1),double(feats_test'),svm_model);
%
% test_scores1 = ws_total'*feats_test;
% test_scores = max([test_scores_straw;test_scores_cup;test_scores_other]);
% test_scores = test_scores_straw+test_scores_cup+0.5*test_scores_other+...
%     .2*double(faceScores_test(sel_test)>-.7)+.2*double(faceDScores_test(sel_test)>-.7)';
test_scores = -test_scores;
[r,ir] = sort(test_scores,'descend');
mm_test = m_test_t(sel_test);
mm_test_toshow = lipImages_test(sel_test);
mm_test_toshow = paintRule(mm_test_toshow,t_test,[],[],1);
% for k = 1:length(mm_test)
%     if (t_test(k))
%         mm_test_toshow{k} = addBorder(mm_test_toshow{k},3,[0 255 0]);
%     end
% end

% figure,imshow(
figure,imshow(multiImage(mm_test_toshow(ir(1:200)),false));
% imwrite(multiImage(mm_test_toshow(ir(1:200)),false),'detections_painted.jpg');
scores_ = -inf*ones(size(test_true_labels));
scores_(sel_test) = test_scores;
%scores_(ib(sel_test)) = test_scores;
p = randperm(length(test_true_labels)); % note that the precision here
% is measured w.r.t the actual number of true instances in the remaining
% selection
[prec,rec,aps] = calc_aps2(scores_(p),test_true_labels(p),sum(test_labels));
figure,plot(rec,prec)
aps


%% exp 8 - extract face images at original scales.
m_test_orig = visualizeLocs2_new(conf,test_ids, qq1_test(4).cluster_locs,'add_border',false,'height',[]);
m_train_orig = visualizeLocs2_new(conf,train_ids, qq1(4).cluster_locs,'add_border',false,'height',[]);


[~,ic,~] = intersect(qq1_test(4).cluster_locs(:,11),1:length(m_test));

% try again using Bag-of-Words on these images.

dict = learnBowDictionary(conf,aaa,true);
model.numSpatialX = [2];
model.numSpatialY = [2];
model.kdtree = vl_kdtreebuild(dict) ;
model.quantizer = 'kdtree';
model.vocab = dict;
model.w = [] ;
model.b = [] ;
model.phowOpts = {};

hists_train = getHists(conf,model,m_train_orig(f_train));
hists_test = getHists(conf,model,m_test_orig(f_test));

y_train = 2 * (t_train(:)==1) - 1;
y_test = 2 * (t_test(:)==1) - 1;

%%
psix_train = vl_homkermap(hists_train, 1, 'kchi2', 'gamma', .5) ;
psix_test = vl_homkermap(hists_test, 1, 'kchi2', 'gamma', .5) ;
% svmModel= svmtrain(y_train(:), double(psix_train'),'-t 2');
% psix_test = vl_homkermap(hists_test, 1, 'kchi2', 'gamma', .5)';
%    dd = psix_test*w;

%%
% [w b info] = vl_pegasos(psix_train,int8(y_train),.00001);
svmModel= svmtrain(y_train, double(psix_train'),'-t 0');
[~, ~, decision_values_train] = svmpredict(zeros(size(psix_train,2),1),double(psix_train'),svmModel);

[r_s,ir_s] = sort(decision_values_train,'ascend');
[mm,~,xx,yy] = multiImage(lipImages_train_s(ir_s(1:150)),false);
figure,imshow(mm)

[~, ~, decision_values_test] = svmpredict(zeros(size(psix_test,2),1),double(psix_test'),svmModel);
%%
close all
total_score = 0*0.001*test_scores(:)+decision_values_test;

[r_s,ir_s] = sort(total_score,'descend');
lipImages_test_s_show = paintRule(lipImages_test_s,t_test);
[mm,~,xx,yy] = multiImage(lipImages_test_s_show(ir_s(1:150)),false);
figure,imshow(mm)

% [prec,rec,aps] = calc_aps2(scores_,test_labels);
[prec,rec,aps] = calc_aps2(total_score,t_test);
% figure,imshow(multiImage(lipImages_test_s(t_test),false))
figure,plot(rec,prec); title(num2str(aps));

%%
[faceLandmarks,allBoxes] = landmarks2struct(landmarks_train);
[faceLandmarks_test] = landmarks2struct(landmarks_test);

%%
ff = find(~train_true_labels);
for q = 1:length(ff)
    clf;
    im = imresize(m_train{ff(q)},2);
    boxes = faceLandmarks(ff(q)).xy;
    if (isempty(boxes))
        continue;
    end
    bc =boxCenters(boxes);
    imshow(im);
    hold on;
    % plotBoxes2(boxes(:,[2 1 4 3]),'color','g');
    % for kk = 1:size(boxes,1)
    %     text(bc(kk,1),bc(kk,2),num2str(kk),'color','r','FontSize',10);
    % end
    
    right_cheek =61:68;
    plot(bc(right_cheek,1),bc(right_cheek,2),'g');
    left_cheek =52:60;
    plot(bc(left_cheek,1),bc(left_cheek,2),'g');
    right_eyebrow = 27:31;
    plot(bc(right_eyebrow,1),bc(right_eyebrow,2),'g');
    left_eyebrow = 16:20;
    plot(bc(left_eyebrow,1),bc(left_eyebrow,2),'g');
    
    cheek = [fliplr(right_cheek),left_cheek];
    plot(bc(cheek,1),bc(cheek,2),'r');
    
    pause(.1);
end


%% get the profiles...
[profiles,valids]= getCheekProfiles(m_train(f_train(t_train)),faceLandmarks(f_train(t_train)));

profiles_1 = profiles(sel_train,:,:);
profiles_pos = profiles_1(t_train,:);
profiles_neg = profiles_1(~t_train,:);

figure,imagesc(profiles_pos)
figure,imagesc(profiles_neg)

figure,plot(mean(profiles_pos))
figure,plot(mean(profiles_neg))

[w,b] = train_classifier(profiles_pos',profiles_neg',.01,10);
%%
[profiles,valids]= getCheekProfiles(m_test_t(f_test(t_test)),faceLandmarks_test(f_test(t_test)));
%%
[profiles_test,valids_test]= getCheekProfiles(m_test_t,faceLandmarks_test);

profiles_test_1 = profiles_test(sel_test,:);
%%
p_scores  =1*w'*profiles_test_1' + test_scores;

[r,ir] = sort(p_scores,'descend');

figure,imshow(multiImage(mm_test_toshow(ir(1:200)),false));

scores_ = p_scores;

[prec,rec,aps] = calc_aps2(scores_(:),t_test);
figure,plot(rec,prec);title(num2str(aps));

%%

for kk = 1:length(ir(1:100))
    [Z0,w,b] = segmentFace_stochastic3(mm_test{ir(kk)},3)
end

getCheekProfiles(m_test_t(ir),faceLandmarks_test(ir));

%% try again matching head templates...
% train_pos_images = multiCrop(m_train(f_train(t_train)),[7 40 50 77]);
train_pos_images = m_train(f_train(t_train));

m_test = visualizeLocs2_new(conf,train_ids,qq1_test(4).cluster_locs,'height',80,'add_border',false,'inflateFactor',2);
m_test = visualizeLocs2_new(conf,test_ids,qq1_test(4).cluster_locs,'height',120,'add_border',false,'inflateFactor',2);
m_test_t = m_test(ic);
conf.features.vlfeat.cellsize = 16;
[r,wsize] = imageSetFeatures2(conf,train_pos_images,true);
conf_d = conf;
conf_d.features.winsize = wsize{1};
[C_,IC_] = vl_kmeans(r,5,'Algorithm','Elkan','NumRepetitions',100);
d_clusters = makeClusterImages(train_pos_images,C_,IC_,r,'drinkingClusters_kmeans1');
conf_d.detection.params.max_models_before_block_method = 1;
conf_d.clustering.num_hard_mining_iters = 12;
d_trained = train_patch_classifier(conf_d,d_clusters,m_train(f_train(~t_train)),'suffix','','toSave',false);
figure,imshow(showHOG(conf_d,d_trained(2)))

test_images = m_test_t(f_test);
%multiCrop(m_test_t(f_test),round(inflatebbox([7 40 50 77],1.5)));
figure,imshow(showHOG(conf_d,d_trained(3)))
figure,imshow(multiImage(test_images))
conf_d.detection.params.detect_min_scale = .8;
[q_test] = applyToSet(conf_d,d_trained,test_images,[],'hog_take2','override',true);


%%
addpath(genpath('/home/amirro/code/3rdparty/proposals'));
train_pos_images = m_train(f_train(t_train));
I = train_pos_images{18};
figure,imshow(I)

train_proposals = generate_all_proposals(m_train);
test_proposals = generate_all_proposals(m_test_t);
save train_proposals.mat train_proposals
save test_proposals.mat test_proposals
% [ranked_regions superpixels image_data] = generate_proposals(I);
%%
close all;
for qq=1:5
    clf;
I = m_train{qq};
ranked_regions = train_proposals(qq).ranked_regions;
superpixels = train_proposals(qq).superpixels;
masks = {};
for kk = 1:length(ranked_regions)
    kk/length(ranked_regions);
     mask = ismember(superpixels, ranked_regions{kk});
%      masks{kk} = mask;
     r = find(~mask);
     red = zeros(size(mask));
     red(r) = 1;
     red = cat(3,red,zeros(size(mask)),zeros(size(mask)));
     mask = cat(3,mask,mask,mask);          
     
     imshow(im2double(mask).*im2double(I)+red);
     pause(.01);
end
end

