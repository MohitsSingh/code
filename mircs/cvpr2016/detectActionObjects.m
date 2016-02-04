if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');
    load('~/storage/misc/images_and_face_obj_full_imdb.mat')
    seg_dir = '~/storage/fra_db_seg_full';
    isTrain = find([fra_db.isTrain]);
    initialized = true;
    %     addpath('~/code/3rdparty/matconvnet-1.0-beta16/matlab');
    %     featureExtractor = DeepFeatureExtractor(conf,true,17);
    addpath('~/code/mircs/cvpr2016/');
    addpath('~/code/mircs/learning');
    addpath('/home/amirro/code/3rdparty/matconvnet-fcn-master/utils');
    %     L = load('~/storage/misc/action_object_pred_full_2_w_classes.mat');
    train_params = struct('classes',1:5,'toBalance',0,'lambdas',.001);
    train_params.task = 'classification';
    train_params.minGroupSize = 0;
    train_params.maxGroupSize = inf;
    train_params.hardnegative = false;
    train_params.standardize=false;
    train_params.minGroupSize=1;
    train_params.maxGroupSize=inf;
    train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
    outPath = '~/storage/fra_action_fcn';
    addpath('~/code/3rdparty/subplot_tight');
    nClasses = 5;
end
%%


showPredsHelper2(fra_db,imdb,300)
imdb.rects = cat(1,fra_db.I_rect);
%% 
train_inds = [training_data.isTrain];
training_data1 = training_data(train_inds);
bestOverlaps = zeros(5,length(fra_db));
has_gt_region = true(size(fra_db));
for t = 1:length(fra_db)
    if none(imdb.labels{t}>=3)
        has_gt_region(t)=false;
    end
end
% 1 2 3 4 5 6 7
% 1 1 3 3 3 3 3
ovp_lut = [1 1 3 3 3 3 3];
for t = 1:length(training_data1)
    t
    %     k = train_inds(t);
    p = training_data1(t);
    if p.is_gt_region,continue,end
    curImgInd = p.imgInd;
    curClassID = fra_db(curImgInd).classID;
    if (~has_gt_region(curImgInd))
        bestOverlaps(curClassID,curImgInd) = nan;
    end
    %     I = imdb.images_data{curImgInd};
    %     F=LL.scores_fine{curImgInd};
    curOvp = bestOverlaps(curClassID,curImgInd);
    ovp = p.ovp_orig_box;
    ovp_types = ovp_lut(p.gt_region_labels);
    f = find(ovp_types==3);
    if any(f)
        %     if length(ovp)==2
        %         ovp = 0;
        %     else
        ovp = ovp(f);
        bestOverlaps(curClassID,curImgInd) = max(bestOverlaps(curClassID,curImgInd),ovp);
    end
    %    ovp = ovp(3);
    %     end    
end
%max_ovps_for_class = zeros(size(data_train));
%%
max_ovps_conv = cell(5,1);
for t = 1:length(bestOverlaps)
    t
    if ~fra_db(t).isTrain,continue,end    
    if bestOverlaps(fra_db(t).classID,t) <.2 0 && has_gt_region(t)
        showPredsHelper2(fra_db,imdb,t);
        figure(5);clf; displayRegions(imdb.images_data{t},imdb.labels{t}>=3,[],'dontPause',true);
        dpc
    end
    max_ovps_conv{fra_db(t).classID}{end+1} = bestOverlaps(fra_db(t).classID,t);   
end
max_ovps_conv = cell2mat(cat(1,max_ovps_conv{:}));
%%
hist(max_ovps_conv',10);
colormap(distinguishable_colors(5));
legend(train_params.classNames );
%%
bar(xo,bsxfun(@rdivide,cumsum(no),sum(no)))
[no_conv,xo] = hist(max_ovps_conv(:),20);
plot(xo,cumsum(no/sum(no)));
no = hist(max_ovps(:),xo);

plot(xo,cumsum(no)/sum(no),'r-');hold on;
plot(xo,cumsum(no_conv)/sum(no_conv),'g-');
legend( {'drinking','smoking','blowing_bubbles','brushing_teeth','phoning'},'Interpreter','none');
% count the o
% bestOverlaps(:,isTrain)
%

%% load the segmentations and find the max,sum of each channel
% just for kicks, load the full high res segmentation too
full_hires_path = '~/storage/fra_action_fcn_hires';
%%
labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};
labels_local_lm = {'none','eye','mouth','chin center','nose tip','face','hand','drink','smoke','blow','brush','phone'};
% summarize probabilities
coarse_sum_features = {};
coase_max_features = {};
fine_sum_features = {};
fine_max_features = {};
for t = 1:length(fra_db)
    t
    %     p = j2m(outPath,fra_db(t));
    %     L = load(p);         
    masked_images = {};
    coarse_probs = LL.scores_coarse{t};
    %fine_probs = LL.scores_fine{t};
    
    p = j2m(full_hires_path,fra_db(t));
    L = load(p);
    fine_probs = L.scores_hires_full;
    fine_probs = bsxfun(@rdivide,exp(fine_probs),sum(exp(fine_probs),3));
    
    
    coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
    coarse_sum = sum(sum(coarse_probs,1),2);
    coarse_max = max(max(coarse_probs,[],1),[],2);
    fine_sum = sum(sum(fine_probs,1),2);
    fine_max = max(max(fine_probs,[],1),[],2);
    coarse_sum_features{t} = coarse_sum(:);
    coase_max_features{t} = coarse_max(:);
    fine_sum_features{t} = fine_sum(:);
    fine_max_features{t} = fine_max(:);
    %     [~,coarse_pred] = max(coarse_probs,[],3);
    %     [~,fine_pred] = max(fine_probs,[],3);
    %     showPredictions(single(imdb.images_data{t}),coarse_pred,coarse_probs,labels_full,1);
    %     showPredictions(single(imdb.images_data{t}),fine_pred,fine_probs,labels_full,2);
    %     dpc
end
%%
%save ~/storage/misc/action_probs.mat coarse_sum_features coase_max_features fine_sum_features fine_max_features
% save ~/storage/misc/action_probs_with_full_fine.mat coarse_sum_features coase_max_features fine_sum_features fine_max_features
%%
load ~/storage/misc/action_probs_with_full_fine.mat
coarse_sum_features = cat(2,coarse_sum_features{:});
coase_max_features =  cat(2,coase_max_features{:});
fine_sum_features =  cat(2,fine_sum_features{:});
fine_max_features =  cat(2,fine_max_features{:});
% get the accuracy for sorting the images by descending order of the
% maximal object prediction in each image.
classes = 1:5;
clear res;
isTest = ~[fra_db.isTrain];
test_labels = [fra_db(isTest).classID];
res_coarse_max = getScorePerformances(classes,test_labels,coase_max_features(4:end,isTest),train_params);
infos_coarse = [res_coarse_max.info];
aps = [infos_coarse.ap]
res_fine_max = getScorePerformances(classes,test_labels,fine_max_features(4:end,isTest),train_params);
infos_fine = [res_fine_max.info];
aps = [infos_fine.ap]

%%
% hkm
% all_feats1 = add_feature(,vl_homkermap(coarse_sum_features,1),'coarse_sum');
my_feats_12 = all_feats(1:2);
my_feats_12(1).feats = my_feats_12(1).feats/700;
my_feats_12(1).abbr = 'G';
my_feats_12(2).feats = my_feats_12(2).feats/700;
my_feats_12(2).abbr = 'F';
N = 1;
all_feats1 = add_feature(my_feats_12,vl_homkermap(coase_max_features,N),'coarse_max_chi2','coarse');
% all_feats1 = add_feature(all_feats1,vl_homkermap(fine_sum_features,1),'fine_sum');
all_feats1 = add_feature(all_feats1,vl_homkermap(fine_max_features,N),'fine_max_chi2','fine');
% all_feats1 = add_feature(all_feats1,vl_homkermap(fine_max_features+coase_max_features,N),'fine_max_chi2','fine_and_coarse');
%% feats1.name = 'funny_feats_coarse';
% feats1.abbr = 'funny_c';
% feats1.feats = funny_feats_coarse;
% fine_max_features+coase_max_features
isTrain = col([fra_db.isTrain]);
valids = true(size(fra_db));
labels = col([fra_db.classID]);
train_params.lambdas = .001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.maxGroupSize = 4;
train_params.minGroupSize = 1;
res_coarse_and_fine = train_and_test_helper(all_feats1,labels,isTrain,valids,train_params);
%%
% res_coarse_and_fine = res;
[summary,sm_coarse_and_fine] = summarizeResults_distributed(res_coarse_and_fine,all_feats1,train_params);
save sm_coarse_and_fine.mat sm_coarse_and_fine
sm_coarse_and_fine
aps = [infos_coarse.ap]
aps = [infos_fine.ap]
% matrix2latex(table2array(sm_coarse_and_fine), 'figures/ours.tex','format','%0.3f','rowLabels',sm_coarse_and_fine.Properties.RowNames,'columnLabels',sm_coarse_and_fine.Properties.VariableNames);
%% show some results.
class_sel = 1;
% test_scores = res_coarse_and_fine{31}.res_test(class_sel).curScores;
% all_test_scores  = cat(1,res_coarse_and_fine{31}.res_test.curScores);
% %imagesc(all_test_scores)
% %
% [~,imageEstClass] = max(all_test_scores,[],1);
% nTotalClasses = 5
% trueClasses = [fra_db(~isTrain).classID];
% CM = confMatrix( trueClasses, imageEstClass, nTotalClasses );
% % % CM = CM(svm.Label,:)
% % train_params.classNames
% imagesc(CM);
% counts = sum(CM,2);
% cm1 = bsxfun(@rdivide,CM,counts);
% imagesc(cm1); colormap jet
% mean(diag(cm1))
% test_scores_coarse = res{22}.res_test(class_sel).curScores;
% test_scores_fine = res{11}.res_test(class_sel).curScores;
% test_scores = test_scores_fine-test_scores_coarse;
showSomeResults(res_coarse_and_fine,outPath,fra_db,imdb,15);
%%
%% find a better way to detect hands ,objects and faces.
clear LL;
LL.scores_coarse = {};
LL.scores_fine = {};
for t = 1:1:length(fra_db);
    %     if fra_db(t).isTrain,continue,end
    t
    p = j2m(outPath,fra_db(t));
    L = load(p);
    coarse_probs = L.scores_full_image;
    fine_probs = L.scores_hires;
    coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
    LL.scores_coarse{t} = coarse_probs;
    LL.scores_fine{t} = fine_probs;
end
feats_interaction = {};
prob_images_coarse = {};
prob_images_fine = {};

%%

%% extract a few candidates from each image
training_data_cell = {};
%%

candidateParams = struct('useRawScores',false,...
    'useLocalMaxima',true,...
    'usePrediction',true,...
    'useMaxObj',false);

object_candidates_dir = '~/storage/misc/obj_candidates';
for t = 1:1:length(fra_db)
    if length(training_data_cell)<t ||  isempty(training_data_cell{t})
        %         t
        %     training_data = getAllObjectCandidates(LL, candidateParams , imdb, fra_db);
        training_data_cell{t} = getAllObjectCandidates(LL, candidateParams , imdb, fra_db,t);
    end
end
%%
showPredsHelper2(fra_db,imdb,t);
%%
%%

save ~/storage/misc/hand_obj_face_training_data_with_regions_localmax_and_pred.mat training_data_cell
%save ~/storage/misc/hand_obj_face_training_data_with_regions_pred_and_max.mat training_data_cell
%%
% load ~/storage/misc/hand_obj_face_training_data_with_regions_localmax.mat
% training_data = cat(2,training_data_cell{:});
%% make a precision_recall curve using the raw scores of the action classes.
isTrain = [fra_db.isTrain];
gt_maps = imdb.labels(isTrain);
raw_results = struct('class_id',{},'map_type',{},'recall',{},'precision',{},'info',{});
n = 0;

map_type = 'coarse';
test_maps = LL.scores_coarse(isTrain);
raw_results = [raw_results,calcRawResults(gt_maps,test_maps,map_type,train_params)];

map_type = 'fine';
test_maps = LL.scores_fine(isTrain);
raw_results = [raw_results,calcRawResults(gt_maps,test_maps,map_type,train_params)];
test_maps = LL.scores_fine(isTrain);
ff = find(isTrain);
for ii = 1:length(ff)
    k = ff(ii);
    test_maps{ii} = test_maps{ii} + LL.scores_coarse{k};
end
map_type = 'fine+coarse';
raw_results = [raw_results,calcRawResults(gt_maps,test_maps,map_type,train_params)];

infos = [raw_results.info];
[ [raw_results.class_id];[infos.ap]]'

inds = reshape([raw_results.class_id],[],3)'
perfs = reshape([infos.ap],[],3)'

save ~/storage/misc/raw_results.mat raw_results

% savefig(fullfile('~/code/mircs/cvpr2016/figures',['coarse_
   
    %%
%% look for images where the action object was not detected.

%%
% first, split the detections per images.
training_per_image = {};
imgs_inds = [training_data.imgInd];
for t = 1:length(fra_db)
    training_per_image{t} = training_data(imgs_inds==t);
end
%% for each image, extract patches using
% 1. the exact segment of the action object
% 2. the big box action object
% 3. the tight bounding box of the action object.
train_inds = [training_data.isTrain];
training_data1 = training_data(train_inds);
bestOverlaps = zeros(5,length(fra_db));
has_gt_region = true(size(fra_db));
for t = 1:length(fra_db)
    if none(imdb.labels{t}>=3)
        has_gt_region(t)=false;
    end
end
%%
% 1 2 3 4 5 6 7
% 1 1 3 3 3 3 3
ovp_lut = [1 2 3 3 3 3 3];
newData_cells = {};
for t = 1:length(fra_db)    
    t
    curTrainingData = training_data_cell{t};
    lenBefore = length(curTrainingData);
    % get boxes and overlaps.    
    curBoxes = cat(1,curTrainingData.bbox_orig);
    I = imdb.images_data{t};
    [a,b,boxArea] = BoxSize(curBoxes);    
    curTrainingData(boxArea < 5) = [];
    lenAfter = length(curTrainingData);
    fprintf('%d(%d)\n',lenAfter,lenBefore);
    curBoxes = cat(1,curTrainingData.bbox_orig);    
    curBoxesBig = cat(1,curTrainingData.bbox);
    % remove boxes which are far away from both hands and faces.    
    localScores = LL.scores_fine{t};
    localScores_coarse = LL.scores_coarse{t};
    ovps_big = cat(1,curTrainingData.ovp_big_box);
    ovps_orig = cat(1,curTrainingData.ovp_orig_box);    
    ovp_types = ovp_lut(curTrainingData(1).gt_region_labels);
    % make sure order is consistent for all samples
    ovps_new_big = zeros(length(curTrainingData),3);
    ovps_new_orig = zeros(length(curTrainingData),3);
    for q = 1:3 
        ff = ovp_types==q;
        if ~any(ff),continue,end
        ovps_new_big(:,q) = ovps_big(:,ovp_types==q);
        ovps_new_orig(:,q) = ovps_orig(:,ovp_types==q);
    end
    ovps_orig = ovps_new_orig;
    ovps_big = ovps_new_big;
    
    % extract features            
    [patches_orig,context_feats_orig_fine,context_feats_orig_max_fine] = extractContextFeatures(I,localScores,curBoxes);
    [patches_big,context_feats_big_fine,context_feats_big_max_fine] = extractContextFeatures(I,localScores,curBoxesBig);
    [~,context_feats_orig_coarse,context_feats_orig_max_coarse] = extractContextFeatures(I,localScores_coarse ,curBoxes);
    [~,context_feats_big_coarse,context_feats_big_max_coarse] = extractContextFeatures(I,localScores_coarse ,curBoxesBig);
%     [~,context_feats_orig_fine,context_feats_orig_max] = extractContextFeatures(I,localScores,curBoxes);
%     [~,context_feats_big,context_feats_big_max] = extractContextFeatures(I,localScores,curBoxesBig);
    
    
    curImgInds = ones(size(curTrainingData))*t;        
    newData = curTrainingData;
    for u = 1:length(newData)
        newData(u).patch_orig = patches_orig{u};
        newData(u).patch_big = patches_big{u};
        newData(u).ovp_orig_box = ovps_orig(u,:);
        newData(u).ovp_big_box = ovps_big(u,:);
        newData(u).context_feats_orig_fine = context_feats_orig_fine{u};
        newData(u).context_feats_orig_max_fine = context_feats_orig_max_fine{u};
        newData(u).context_feats_big_fine = context_feats_big_fine{u};
        newData(u).context_feats_big_max_fine = context_feats_big_max_fine{u};
        newData(u).context_feats_orig_coarse = context_feats_orig_coarse{u};
        newData(u).context_feats_orig_max_coarse = context_feats_orig_max_coarse{u};
        newData(u).context_feats_big_coarse = context_feats_big_coarse{u};
        newData(u).context_feats_big_max_coarse = context_feats_big_max_coarse{u};

    end
    newData_cells{t} = newData;
end
%%
save ~/storage/misc/newData_cells2.mat newData_cells
%%
%%
% remove all boxes outside of person rectangles
% sel_ = {};
% for t = 1:length(newData_cells)
%     curData = newData_cells{t};
%     curRect = imdb.rects(t,:);
%     curBoxes = cat(1,curData.bbox_orig);
%     [ovp,int] = boxesOverlap(curBoxes,curRect);
%     [~,~,a] = BoxSize(curBoxes);
%     sel_{t} = int>.7*a;
% %     sel_{t} = true(size(int));%int>.7*a;
% end   
% sel_ = cat(1,sel_{:});
patches = cellfun3(@(x) {x.patch_orig},newData_cells,2);
patches_big = cellfun3(@(x) {x.patch_big},newData_cells,2);
context_feats_orig_fine = cellfun3(@(x) {x.context_feats_orig_fine},newData_cells,2);
context_feats_orig_max_fine = cellfun3(@(x) {x.context_feats_orig_max_fine},newData_cells,2);
context_feats_big_fine = cellfun3(@(x) {x.context_feats_big_fine},newData_cells,2);
context_feats_big_max_fine = cellfun3(@(x) {x.context_feats_big_max_fine},newData_cells,2);
context_feats_orig_coarse = cellfun3(@(x) {x.context_feats_orig_coarse},newData_cells,2);
context_feats_orig_max_coarse = cellfun3(@(x) {x.context_feats_orig_max_coarse},newData_cells,2);
context_feats_big_coarse = cellfun3(@(x) {x.context_feats_big_coarse},newData_cells,2);
context_feats_big_max_coarse = cellfun3(@(x) {x.context_feats_big_max_coarse},newData_cells,2);


FF = {context_feats_orig_fine;...
context_feats_orig_max_fine;...
context_feats_big_fine;...
context_feats_big_max_fine;...
context_feats_orig_coarse;...
context_feats_orig_max_coarse;...
context_feats_big_coarse;...
context_feats_big_max_coarse};

% FF = {;...
% context_feats_orig_max_fine;...
% ;...
% context_feats_big_max_fine;...
% ;...
% context_feats_orig_max_coarse;...
% ;...
% context_feats_big_max_coarse};


for t = 1:length(FF)
    t
   FF{t} = vl_homkermap(cellfun3(@col,FF{t},2),1);
end

context_feats = cat(1,FF{:});

% context_feats = cellfun3(@col,context_feats,2);
cur_train_inds = cellfun3(@(x) [x.isTrain],newData_cells,2);
all_ovps = cellfun3(@(x) cat(1,x.ovp_orig_box),newData_cells,1);
all_ovp_big = cellfun3(@(x) cat(1,x.ovp_big_box),newData_cells,1);
all_boxes = cellfun3(@(x) cat(1,x.bbox_orig),newData_cells,1);
all_boxes_big = cellfun3(@(x) cat(1,x.bbox),newData_cells,1);
all_img_inds = cellfun3(@(x) cat(1,x.imgInd),newData_cells,1);

% patches = patches(sel_);
% patches_big = patches_big(sel_);
% context_feats = context_feats(:,sel_);
% cur_train_inds = cur_train_inds(sel_);
% all_ovps = all_ovps(sel_,:);
% all_boxes = all_boxes(sel_,:);
% all_img_inds = all_img_inds(sel_);
cur_train_params = train_params;
cur_train_params.classes = [0 1];
cur_train_params.classNames = {'none','object'};
sel_ = 3;
sel_pos = all_ovps(:,sel_) > .5;
sel_neg = all_ovps(:,sel_) <= .2;
curLabels = zeros(size(all_ovps,1),1);
curLabels(sel_pos)=1;
curLabels(sel_neg)=0;
%
% 
% for t = 1:10:end
%     m = all_img_inds(t);
% % %     I = imdb.images_data(all_boxes
% end
% s
%%


% check patch size 

% sizes_orig = cellfun3(@size2,patches,1);

patch_appearance = featureExtractor.extractFeaturesMulti(patches);
save ~/storage/misc/patch_appearance.mat patch_appearance 
patch_big_appearance = featureExtractor.extractFeaturesMulti(patches_big);
save ~/storage/misc/patch_big_appearance.mat patch_big_appearance 

%%
% find the best per-image overlap
% get the ground-truth bounding box for all images.
gt_boxes = zeros(length(fra_db),4);
gt_boxes_big = zeros(length(fra_db),4);
for t = 1:length(fra_db)
    B = imdb.labels{t}>=3;
    if (any(B(:)))
        gt_boxes(t,:) = region2Box(B);
        gt_boxes_big(t,:) = inflatebbox(gt_boxes(t,:),[size(B,1)/3],'both',true);
    end
end
[a,b,c] = BoxSize(gt_boxes);
% figure,plot(c.^.5)
%%
[overlaps1,inds1] = analyzeOverlaps(all_img_inds,all_boxes,gt_boxes,imdb.images_data);

[overlaps,inds] = analyzeOverlaps(all_img_inds,all_boxes_big,gt_boxes_big,imdb.images_data);
%%
[r,ir] = sort(overlaps,'ascend');
for u = 1:1:length(r)
    r(u)
    t = ir(u);
    if all(gt_boxes(t,:)==0),continue,end
%     if t~=377,continue,end
    if isTrain(t),continue,end
    I = imdb.images_data{t};
    figure(10); clf; imagesc2(I);
    plotBoxes(all_boxes(all_img_inds==t,:),'r-','LineWidth',3);
    plotBoxes(all_boxes(inds(t),:),'g-','LineWidth',1);
    plotBoxes(gt_boxes(t,:),'m-','LineWidth',2);
          
%     showPredsHelper2(fra_db,imdb,t);
    dpc
end

%%
% {fra_db.imageID}
% now show an overlap histogram for each class.
labels = [fra_db.classID];
isTrain = [fra_db.isTrain];
[overlaps,inds] = analyzeOverlaps(all_img_inds,all_boxes,gt_boxes);
%%
figure(1);clf
for t = 1:5
    [no,xo] = hist(overlaps(labels==t & ~isTrain),[0:.1:1]);
    no = no/sum(no);
    subplot(3,2,t); bar(xo,no);ylim([0 1]);
    title(train_params.classNames{t});
end
%%
[overlaps2,inds] = analyzeOverlaps(all_img_inds,inflatebbox(all_boxes,1.5,'both',false),gt_boxes);
% {fra_db.imageID}
% now show an overlap histogram for each class.
figure(2);clf
for t = 1:5
    [no,xo] = hist(overlaps2(labels==t & ~isTrain),[0:.1:1]);
    no = no/sum(no);
    subplot(3,2,t); bar(xo,no);ylim([0 1]);
    title(train_params.classNames{t});
end

%% 

%%
%% 
% feats_patches = {};
good_ovps = all_ovps(:,3)>.5;
isTrain_1 = [fra_db(all_img_inds).isTrain];
all_labels = [fra_db(all_img_inds).classID];
all_labels(all_ovps(:,3)<.5) = 0;
train_params1 = train_params;
train_params1.classes = 1;
train_params1.classNames = 'Obj';
all_labels(all_labels>0) = 1;


% feats_patches1 = cat(2,feats_patches{:});
%%
%feats_r1 =[patch_appearance/700;patch_big_appearance/700;vl_homkermap(context_feats,1)];
% feats_r1 =[context_feats,1)];
%%
% train_params1.lambdas = [.1 .01 .001 .0001]
train_params1.lambdas = [.001];
% feats_r1 =[patch_appearance/1000;vl_homkermap(context_feats,1)];
res_train = train_classifiers(context_feats(:,isTrain_1),all_labels(isTrain_1),[],train_params1);
%
res_test = apply_classifiers(res_train,context_feats(:,~isTrain_1),all_labels(~isTrain_1),train_params1);
infos = [res_test.info];
[infos.ap]
%%

%% 
% train_params1.lambdas = [.1 .01 .001 .0001]
% feats_r1 =[patch_appearance/1000;vl_homkermap(context_feats,1)];
res_train = train_classifiers(feats_r1(:,isTrain_1),all_labels(isTrain_1),[],train_params1);
%
res_test = apply_classifiers(res_train,feats_r1(:,~isTrain_1),all_labels(~isTrain_1),train_params1);
infos = [res_test.info];
[infos.ap]

%% do it as a regression task
train_params1.task = 'regression';
train_params1.lambdas = .001;
res_train = train_classifiers(feats_r1(:,isTrain_1),all_labels(isTrain_1),all_ovps(isTrain_1,3),train_params1);

%%
res_test = apply_classifiers(res_train,feats_r1(:,~isTrain_1),all_labels(~isTrain_1),train_params1);
infos = [res_test.info];
[infos.ap]

%%
%% find the best action object region in each image, and from this infer the actual action object...

% [values,inds] = splitToGroups(,all_inds)

s = res_train(1).classifier_data.w(1:end-1)'*feats_r1;

[r,ir] = sort(s,'descend');
image_visited = zeros(size(fra_db));

showEachImageOnce = true;
debug_factor = 0;
showTrainingImages = false;
showTestingImages = true;
showAnything = false;
[r,ir] = sort(s,'descend');
% curValids = true(size(fra_db));

chosenScores=  {};
for it = 1:1:length(s)
    r(it)
    k = ir(it);
    imgInd = all_img_inds(k);
    if isTrain(imgInd)
        if ~showTrainingImages,continue,end
    else
        if ~showTestingImages,continue,end
    end
%     if showTrainingImages && ~isTrain(imgInd),continue,end
    if ~isempty(img_sel) && imgInd~=img_sel,continue,end
    if showEachImageOnce
        if (image_visited(imgInd)),continue,end
    end
    image_visited(imgInd)=1;
    nImagesVisited = nnz(image_visited);
    nImagesVisited                
%     if isTrain(imgInd),continue,end
    I = imdb.images_data{imgInd};
    patch_of_img{imgInd} = cropper(I,all_boxes(k,:));
    patch_of_img_bigger{imgInd} = cropper(I,round(inflatebbox(all_boxes(k,:),2,'both',false)));
    chosenFeats{imgInd} = feats_r1(:,k);
    chosenScores{imgInd} = r(it);
        if debug_factor ~=0 && mod(nImagesVisited,debug_factor)~=0,continue,end
    if ~showAnything,continue,end
    figure(10); clf;imagesc2(I);
    plotBoxes(all_boxes(k,:));
%     showPredsHelper2(fra_db,imdb,imgInd);
    dpc
end
%%
patches_feats = featureExtractor.extractFeaturesMulti(patch_of_img);
patches_feats_bigger = featureExtractor.extractFeaturesMulti(patch_of_img_bigger);


%%
ZZ = {};
for iClass = 1:5
%     vl_tightsubplot(5,1,iClass)
    sel_ = ~isTrain & labels==iClass;
    curClassScores = [chosenScores{sel_}]
    curPatches = patch_of_img(sel_);
    [v,iv] = sort(curClassScores,'descend');
%     iv = iv(1:10);    
    ZZ = [ZZ,curPatches((1:5))];
%     for u = 1:length(iv)
%         vl_tight(subplot(iClass,
end
% end
x2(ZZ)
    
%x2(patch_of_img(~isTrain & labels==1))
%

%%
DD = 700;
% DD=1
% all_feats_hof = add_feature(all_feats1,[hand_feats;face_feats;obj_feats]/DD,'hand_feats','H1');
% cur_feats_to_add = patches_feats;
%all_feats_soft = add_feature(all_feats1([1 2 3 4]),(patches_seg_feats+patches_hard_feats+patches_feats)/(DD),'obj');
% all_feats_with_boxes = add_feature(all_feats1([1 2 3 4]),cat(2,patches_feats{:})/DD,'obj_box');
all_feats_with_boxes = add_feature(all_feats1,cat(2,patches_feats{:})/DD+cat(2,patches_feats_bigger{:})/DD,'obj_box_big');
%
isTrain = [fra_db.isTrain];
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 1;
train_params.maxGroupSize = 1;
%%
res_with_boxes = train_and_test_helper(all_feats_with_boxes,labels(:),isTrain(:),valids_masks,train_params);
% for p = 1:length(patches)
%     if ~isTrain(p);
%         clf; imagesc2(mImage(patches{p}));
%         dpc
%     end
% end
%
[summary,sm_boxes] = summarizeResults(res_with_boxes,all_feats_with_boxes,train_params);
sm_soft
%%
%%
train_params.minGroupSize = 4;
train_params.maxGroupSize = 5;
res_boxes1 = train_and_test_helper(all_feats_with_boxes,labels(:),isTrain(:),valids_masks,train_params);
%%
[summary,sm_boxes1] = summarizeResults_distributed(res_boxes1,all_feats_with_boxes,train_params);



%% visualize the results a bit..
s = res_train(1).classifier_data.w(1:end-1)'*feats_r1;
[scores,inds] = splitToGroups(s,all_img_inds);
[ovps_gt,inds_gt] = splitToGroups(all_ovps(:,3),all_img_inds);
%%
% now show for each image the actual box selected
patches = {};
patches_prob = {};
patches_hard_mask = {};
patches_soft_mask = {};
patches_seg_mask = {};
valids_masks = true(size(fra_db));



%%
P = {};
for t = 1:length(patches_seg_mask)
    t
    pp = patches_soft_mask{t};
    if iscell(pp)
        P{t} = pp{1};
    else
        P{t} = pp;
    end
end
% % P = cellfun2(@(x)  x{1}, patches_soft_mask);
% x2(P(~isTrain))
%
%%
ZZ = {};
for iClass = 1:5
    %     vl_tightsubplot(5,1,iClass)
    sel_ = ~isTrain & labels==iClass;
    curClassScores = [chosenScores{sel_}]
    curPatches = P(sel_);
    [v,iv] = sort(curClassScores,'descend');
    %     iv = iv(1:10);    
    z1 = 4;z2 = z1+21;
    ZZ = [ZZ,curPatches((z1:5:z2))];
    %     for u = 1:length(iv)
    %         vl_tight(subplot(iClass,
end
x2(ZZ);
%% compute the average max OVP of big,orig for each rank.
VVV = 20
ovps = zeros(VVV,length(fra_db));
% for t = 1:10
for u = 1:length(fra_db)
    u
    v = scores{u};
    [v,iv] = sort(v,'descend');
    curBox = all_boxes_big(inds{u}(iv),:);
    pick = nms([curBox col(v(1:length(iv)))], .7);
    pick = pick(1:min(length(pick),VVV));
    curBox = curBox(pick,:);
    z = length(pick);
    ovp = boxesOverlap(curBox,gt_boxes_big(u,:));
    ovps(1:z,u) = ovp;
    
end

ovps = ovps(:,~isTrain);

plot(mean(cummax(ovps),2))

%% 


s = res_train(1).classifier_data.w(1:end-1)'*feats_r1;
[scores,inds] = splitToGroups(s,all_img_inds);
[ovps_gt,inds_gt] = splitToGroups(all_ovps(:,3),all_img_inds);
avg_img = featureExtractor.net.normalization.averageImage;
% get training patches
avg_img_u = uint8(avg_img);
box_ovps = {};
box_classes = {};
is_train = {};
for u = 1:1:length(fra_db)
    u
    if (mod(u,5)==0)
        disp(u)
    end
    I = imdb.images_data{u};
    S = LL.scores_coarse{u}+LL.scores_fine{u};        
    S = max(S(:,:,4:end),[],3);
    if isTrain(u)
        disp('isTrain')
        v = ovps_gt{u};
        S1 = imdb.labels{u}>=3;        
        if (none(S1))
            valids_masks(u) = false;
            [patches{u},patches_prob{u},patches_hard_mask{u},patches_soft_mask{u},patches_seg_mask{u}] = ...
                deal(avg_img_u,avg_img_u,avg_img_u,avg_img_u,avg_img_u);
            continue;
        end
        [v,iv] = sort(v,'descend');
        iv = iv(1);                                
        curBox = all_boxes_big(inds{u},:);
        curOvps = boxesOverlap(curBox,gt_boxes_big(u,:));
        [v,iv] = sort(curOvps,'descend');       
        curBox = [curBox(iv,:) v];
        pick = nms(curBox,.7);
        curBox = curBox(pick,:);
        v = v(pick);
        sel_ = v >=.5 | v<=.1;
        curBox = curBox(sel_,:);
        v = v(sel_);       
        box_ovps{u} = v;
        is_train{u} = true(size(box_ovps{u}));        
        box_classes{u} = ones(size(box_ovps{u}))*fra_db(u).classID;
%         [patches{u},patches_prob{u},patches_hard_mask{u},patches_soft_mask{u},patches_seg_mask{u}] = cropAndMask(I,S,curBox,avg_img);
        
%         
%         figure(1); clf; imagesc2(I);
%         figure(2); curPatches = patches{u};
%         curPatches = paintRule(curPatches,v>=.5,[0 255 0],[255 0 0],3);
%         clf; imagesc2(mImage(curPatches));
%         dpc
        
    else
                disp('isTest')
        %          continue
        v = scores{u};
        [v,iv] = sort(v,'descend');
        %         iv = iv(1:min(length(iv),10));
        curBox = all_boxes_big(inds{u}(iv),:);
        curOvps = boxesOverlap(curBox,gt_boxes_big(u,:));                        
        pick = nms([curBox col(v(1:length(iv)))], .7);
%         pick = pick(1:min(3,length(pick)));
        curBox = curBox(pick,:);
        curOvps = curOvps(pick);
        [curOvps,iz] = sort(curOvps,'descend');        
        curBox = curBox(iz,:);
        box_ovps{u} = curOvps;
        is_train{u} = false(size(box_ovps{u}));
        box_classes{u} = ones(size(box_ovps{u}))*fra_db(u).classID;
        %         clf; imagesc2(I);
        %         colors = {'r-','g-','b-'};
        %         for uuu = 1:min(3,length(pick))
        %             plotBoxes(curBox(uuu,:),colors{uuu},'LineWidth',2);
        %         end
        %         dpc;continue               
        
        
        
%         figure(1); clf; imagesc2(I);
%         figure(2); curPatches = patches{u};
%         curPatches = paintRule(curPatches,curOvps>=.5,[0 255 0],[],3);
%         curPatches = paintRule(curPatches,curOvps<=.2,[255 0 0],[],3);
%         clf; imagesc2(mImage(curPatches));
%         
%         dpc
    end
    
    
    [patches{u},patches_prob{u},patches_hard_mask{u},patches_soft_mask{u},patches_seg_mask{u}] = cropAndMask(I,S,curBox,avg_img);
    
    
    continue;
    mm = 2;nn = 3;
    clf; vl_tightsubplot(mm,nn,1); imagesc2(I);
    plotBoxes(curBox);
    vl_tightsubplot(mm,nn,2); imagesc2(sc(cat(3,S,im2single(I)),'prob_jet'));
    subplot(mm,nn,3); imagesc2(curPatch);
    subplot(mm,nn,4); imagesc2(patch_hard/255);
    subplot(mm,nn,5); imagesc2(patch_soft/255);
    subplot(mm,nn,6);
    imagesc2(patch_soft2/255);
    dpc
end
%%
%% 
%%

all_patch_ovps = cellfun3(@col,box_ovps(valids_masks),1);
patch_img_inds = {};
for t = 1:length(box_classes)
%     box_classes{t} = (ones(size(box_classes{t}))*fra_db(t).classID).*(box_ovps{t}>.5);
    patch_img_inds{t} = ones(size(box_classes{t}))*t;
end
all_patch_labels = cellfun3(@col,box_classes(valids_masks),1);
all_patch_isTrain = cellfun3(@col, is_train(valids_masks),1)>0;
patch_img_inds = cat(1,patch_img_inds{(valids_masks)});
% now try to differentiate the action classes based on patches alone...
% DD=1
% all_feats_hof = add_feature(all_feats1,[hand_feats;face_feats;obj_feats]/DD,'hand_feats','H1');

all_patch_feats = cat(2,patches_feats{(valids_masks)});

% cur_feats_to_add = patches_feats;
%all_feats_soft = add_feature(all_feats1([1 2 3 4]),(patches_seg_feats+patches_hard_feats+patches_feats)/(DD),'obj');
patch_feats1 = add_feature([],all_patch_feats,'p');
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 1;
train_params.maxGroupSize = 1;
%%
res_p1 = train_and_test_helper(patch_feats1(end),all_patch_labels(:),all_patch_isTrain>0, [] ,train_params);

scores1 = res_p1{1}.res_test(1).curScores;

all_patches = cat(2,patches{valids_masks});


[r,ir] = sort(scores1,'descend');

test_patch_ovps = all_patch_ovps(~all_patch_isTrain);
test_patches = all_patches(~all_patch_isTrain);
[r,ir] = sort(scores1,'descend');
% [r,ir] = sort(test_patch_ovps,'descend');
displayImageSeries2(test_patches(ir));


[r,ir] = sort(all_patch_ovps,'descend');
displayImageSeries2(all_patches(ir));

% 
% test_inds = patch_img_inds(~all_patch_isTrain);
% [values,inds] = splitToGroups(scores1,test_inds);
% 


% for p = 1:length(patches)
%     if ~isTrain(p);
%         clf; imagesc2(mImage(patches{p}));
%         dpc
%     end
% end
%
[summary,sm1] = summarizeResults(res_p1,patch_feats1,train_params);
sm1
%%


edit train_and_test_helper

% patches_feats = {};
% patches_soft_feats = {};
% patches_hard_feats = {};

patches_feats = featureExtractor.extractFeaturesMulti(patches);
patches_soft_feats = featureExtractor.extractFeaturesMulti(patches_soft_mask);
patches_hard_feats = featureExtractor.extractFeaturesMulti(patches_hard_mask);
patches_seg_feats = featureExtractor.extractFeaturesMulti(patches_seg_mask);

% 
% valids_masks = cellfun(@(x) ~isempty(x),patches_seg_mask);
% 
% for t = 1:length(valids_masks)
%     if ~valids_masks(t)
%         patches_seg_mask{t} = avg_img;
%     end
% end

save('~/storage/misc/patches_soft_hard_full.mat',...
    'patches', 'patches_soft_mask', 'patches_hard_mask', 'patches_seg_mask',...
    'patches_feats', 'patches_soft_feats', 'patches_hard_feats', 'patches_seg_feats');
%%
DD = 700;
% DD=1
% all_feats_hof = add_feature(all_feats1,[hand_feats;face_feats;obj_feats]/DD,'hand_feats','H1');
cur_feats_to_add = patches_feats;
%all_feats_soft = add_feature(all_feats1([1 2 3 4]),(patches_seg_feats+patches_hard_feats+patches_feats)/(DD),'obj');
all_feats_soft = add_feature(all_feats1([1 2 3 4]),cellfun2(@(x) x(:,1:min(3,size(x,2)))/DD,patches_feats),'obj');
%
isTrain = [fra_db.isTrain];
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 1;
train_params.maxGroupSize = 1;
%%
res_soft = train_and_test_helper(all_feats_soft(end),labels(:),isTrain(:),valids_masks,train_params);

for p = 1:length(patches)
    if ~isTrain(p);
        clf; imagesc2(mImage(patches{p}));
        dpc
    end
end

%
[summary,sm_soft] = summarizeResults(res_soft,all_feats_soft(end),train_params);
sm_soft
%%
%%
train_params.minGroupSize = 4;
train_params.maxGroupSize = 5;
res_soft1 = train_and_test_helper(all_feats_soft,labels(:),isTrain(:),valids_masks,train_params);
%%
[summary,sm_soft1] = summarizeResults_distributed(res_soft1,all_feats_soft,train_params);
save ~/storage/misc/results_all.mat res_soft res_soft1 sm_soft sm_soft1 all_feats_soft

matrix2latex(table2array(sm_soft), 'figures/sm_soft.tex','format','%0.2f','rowLabels',sm_soft.Properties.RowNames,'columnLabels',sm_soft.Properties.VariableNames);
matrix2latex(table2array(sm_soft1), 'figures/sm_soft1.tex','format','%0.2f','rowLabels',sm_soft1.Properties.RowNames,'columnLabels',sm_soft1.Properties.VariableNames);
% 

% show some nice results...
showSomeResults(res_soft1,outPath,fra_db,imdb,6,1); 

%%
% compare the whole patch, soft, hard, seg approaches
DD = 1;
various_patch_features = add_feature([],patches_feats/DD,'patch');
various_patch_features = add_feature(various_patch_features,patches_soft_feats/DD,'soft');
various_patch_features = add_feature(various_patch_features,(patches_hard_feats)/DD,'hard');
various_patch_features = add_feature(various_patch_features,patches_seg_feats/DD,'seg');
various_patch_features = add_feature(various_patch_features,(patches_soft_feats+patches_hard_feats+patches_feats)/(3*DD),'seg1');
train_params.minGroupSize = 1;
train_params.maxGroupSize = 1;
train_params.lambdas = .001;
res_various = train_and_test_helper(various_patch_features,labels(:),isTrain(:),valids_masks(:),train_params);
[~,sm_various] = summarizeResults_distributed(res_various,various_patch_features,train_params);
sm_various
% x2(patches_prob(isTrain))



%%
%% find the best action object region in each image, and from this infer the actual action object...
% [values,inds] = splitToGroups(,all_inds)
[r,ir] = sort(s,'descend');
image_visited = zeros(size(fra_db));

showEachImageOnce = true;
debug_factor = 20;
showTrainingImages = false;
showTestingImages = true;
showAnything = true;
[r,ir] = sort(s,'descend');
% curValids = true(size(fra_db));
for it = 1:1:length(s)
    r(it)
    k = ir(it);
    imgInd = all_img_inds(k);
    if isTrain(imgInd)
        if ~showTrainingImages,continue,end
    else
        if ~showTestingImages,continue,end
    end
    if showTrainingImages && ~isTrain(imgInd),continue,end
    if ~isempty(img_sel) && imgInd~=img_sel,continue,end
    if showEachImageOnce
        if (image_visited(imgInd)),continue,end
    end
    image_visited(imgInd)=1;
    nImagesVisited = nnz(image_visited);
    nImagesVisited
        
    
    
%     if isTrain(imgInd),continue,end
    I = imdb.images_data{imgInd};
    patch_of_img{imgInd} = cropper(I,all_boxes(k,:));
    chosenFeats{imgInd} = feats_r1(:,k);
        if debug_factor ~=0 && mod(nImagesVisited,debug_factor)~=0,continue,end
    if ~showAnything,continue,end
    figure(10); clf;imagesc2(I);
    plotBoxes(all_boxes(k,:));
%     showPredsHelper2(fra_db,imdb,imgInd);
    dpc
end
%%

% choose the highest scoring action object in each image as the action 
% object candidate, 
s = res_train(1).classifier_data.w(1:end-1)'*feats_r1;
[r,ir] = sort(s,'descend');
image_visited = zeros(size(fra_db));
patch_of_img = {};
chosenFeats = {};
img_sel = [];
[scores,inds] = splitToGroups(s,all_img_inds);
chosenFeats = {};
for t = 1:1:length(inds)
%     if t~=1,continue,end
    curScores = scores{t};
    curInds = inds{t};
    curBoxes = all_boxes(curInds,:);
    [v,iv] = sort(curScores,'descend');
    iv = iv(1:min(3,length(iv)));
    if isTrain(t)
        chosenFeats{t} = feats_r1(:,curInds(iv(1)));
    else
        chosenFeats{t} = feats_r1(:,curInds);
    end
    
    
%             [v,iv] = sort(curScores,'descend');
%         clf; imagesc2(I); plotBoxes(curBoxes,'g-','LineWidth',3);
%         plotBoxes(curBoxes(iv(1),:),'r-','LineWidth',1);
end

%%

%%
for u = 1:length(chosenFeats)
    if ~image_visited(u)
        chosenFeats{u} = zeros(size(chosenFeats{1}));
    end
end
% P = patch_of_img(~isTrain);
P = patch_of_img;
for u = 1:length(P)
    if isempty(P{u})
        P{u} = zeros(50,50,3);
    end
end

% x2(P(~isTrain & labels==2))

%%
cur_train_params.lambda = .01;
interaction_feats_new = add_feature([],context_feats,'context_feats','context_feats');
interaction_feats_new = add_feature(interaction_feats_new,patch_appearance,'p_app','p_app');
cur_train_params.minGroupSize=1;
res_interaction_feats_new = train_and_test_helper(interaction_feats_new,...
    curLabels,...
    cur_train_inds,[],cur_train_params);
%
[~,sm_interaction] = summarizeResults_distributed(res_interaction_feats_new,interaction_feats_new,cur_train_params);
sm_interaction
%%
cur_test_inds = find(~cur_train_inds);
test_img_inds = all_img_inds(cur_test_inds);
test_scores = res_interaction_feats_new{1}.res_test(2).curScores;
%% get 
[r,ir] = sort(test_scores,'descend');

maxBoxesPerImage = 3;
image_visited = zeros(size(fra_db));
n = 0;
boxes_for_image = zeros(length(fra_db),maxBoxesPerImage,4);
for it = 1:length(ir)
    k = ir(it);
    curImgInd = test_img_inds(k);
    curBox = all_boxes(cur_test_inds(k),:);
    [ovp,int] = boxesOverlap(curBox,imdb.rects(curImgInd,:));
    [~,~,a] = BoxSize(curBox);
    if int<a/2,continue,end    
    if image_visited(curImgInd)>maxBoxesPerImage-1,continue,end
    image_visited(curImgInd) = image_visited(curImgInd)+1;
    n = n+1
    I = imdb.images_data{curImgInd};    
    boxes_for_image(curImgInd,image_visited(curImgInd),:) = curBox;
%     clf; imagesc2(I); plotBoxes(curBox);
%     dpc(.01)
end

%% 
DD = 1000;
% DD=1
all_feats_hof = add_feature(all_feats1,chosenFeats,'proxy','proxy');

% for q = 1:length(all_feats_hof)
%     f = all_feats_hof(q).feats;
%     if (size(f,1)==4096)
%         f(f<0) = 0;
%         all_feats_hof(q).feats = f;
%     end
% end

isTrain = [fra_db.isTrain];
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .0001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 1;
train_params.maxGroupSize= 1;
res_hof_single = train_and_test_helper(all_feats_hof,labels(:),isTrain(:),valids(:),train_params);
%%
% res_coarse_and_fine = res;
[summary,sm_hof_single] = summarizeResults_distributed(res_hof_single,all_feats_hof,train_params);
% x2(face_imgs(1:50:end));
% save summary_single.mat sm_hof_single
% end
%%
train_params.lambdas = .001;
train_params.minGroupSize = 4;
train_params.maxGroupSize= inf;
res_hof_one_out = train_and_test_helper(all_feats_hof,labels(:),isTrain(:),valids(:),train_params);
% 
% train_params.lambdas = .001;
% train_params.minGroupSize =4;
% res_hof_one_out1 = train_and_test_helper(all_feats_hof,labels(:),isTrain(:),valids(:),train_params);
% [summary,sm_hof_one_out1] = summarizeResults_distributed(res_hof_one_out1,all_feats_hof,train_params);

%%
% res_coarse_and_fine = res;
[summary,sm_hof_one_out] = summarizeResults_distributed(res_hof_one_out,all_feats_hof,train_params);
% save summary_one_out.mat sm_hof_one_out
%%
%%
%showSomeResults(res_hof_one_out,outPath,fra_db,imdb,5,1); 
showSomeResults(res_hof_one_out,outPath,fra_db,imdb,6,1); 
%%
t = 1051
curScores = scores{t};
curInds = inds{t};
curBoxes = all_boxes(curInds,:);
[v,iv] = sort(curScores,'descend');
iv = iv(1:min(3,length(iv)));
I = imdb.images_data{t};
clf; imagesc2(I); plotBoxes(curBoxes,'g-','LineWidth',3);
plotBoxes(curBoxes(iv(1),:),'r-','LineWidth',1);

%%

% objects_boxed_train = objects_boxed([fra_db.isTrain]);
% res_train = train_classifiers(train_features,train_labels,[],train_params,valids(isTrain));
% objects_boxed_test = {};
image_object_patches = {};
Z = 0;
for t = 1:1:length(fra_db)
    if fra_db(t).isTrain,continue,end
    Z = Z+1;
    if mod(Z,30)~=0,continue,end
    t
    
    I = imdb.images_data{t};
    curBoxes = squeeze(boxes_for_image(t,:,:));
    curBoxes = reshape(curBoxes,[],4);
    clf; imagesc2(I); plotBoxes(curBoxes );
    dpc
end
%%
for t = 1:10:length(fra_db)
    if fra_db(t).isTrain,continue,end
    t
    I = imdb.images_data{t};
    clf; imagesc2(I); plotBoxes(squeeze(boxes_for_image(t,:,:)));
    dpc
end
%%
for t = 1:length(training_data1)
    t
    %     k = train_inds(t);
    p = training_data1(t);
    if p.is_gt_region,continue,end
    curImgInd = p.imgInd;
    curClassID = fra_db(curImgInd).classID;
    if (~has_gt_region(curImgInd))
        bestOverlaps(curClassID,curImgInd) = nan;
    end
    %     I = imdb.images_data{curImgInd};
    %     F=LL.scores_fine{curImgInd};
    curOvp = bestOverlaps(curClassID,curImgInd);
    ovp = p.ovp_orig_box;
    ovp_types = ovp_lut(p.gt_region_labels);
    f = find(ovp_types==3);
    if any(f)
        %     if length(ovp)==2
        %         ovp = 0;
        %     else
        ovp = ovp(f);
        bestOverlaps(curClassID,curImgInd) = max(bestOverlaps(curClassID,curImgInd),ovp);
    end
    %    ovp = ovp(3);
    %     end    
end
%%
% (see some stats about ovps later)
% now find the amount of box overlap per object.
%%
img_box_ovp = zeros(length(fra_db));
img_region_ovp = zeros(length(fra_db));
for t = 1:length(fra_db)
    t
    if fra_db(t).isTrain,continue,end
    curRecords = training_per_image{t};
    toKeep = ~[curRecords.is_gt];
    region_ovps = cat(1,curRecords(toKeep).ovp_region);
    box_ovps = cat(1,curRecords(toKeep).ovp_orig_box);
    boxes = cat(1,curRecords(toKeep).bbox_orig);
    if (size(box_ovps,2)>2)
        box_ovps = box_ovps(:,3);
        [box_ovps,imax_box] = max(box_ovps,[],1);
        [region_ovps,imax_region] = max(region_ovps(:,3),[],1);
        if box_ovps < .5
            clf; imagesc2(imdb.images_data{t});
            plotBoxes(boxes(imax_box,:));
            plotBoxes(boxes(imax_region,:));
            [t box_ovps region_ovps(imax_box,3)]
            dpc
        end
    end
end
%% extract various "interaction" features from these : maximal and mean-pooled features.
max_vecs_fine = {};
pred_vecs = {};
for t = 1:length(training_data)
    max_vecs_fine{t} = squeeze(max(max(training_data(t).feats_fine,[],1),[],2));
    [~,curPredVec] = max(training_data(t).feats_fine,[],3);
    %z = zeros(size(training_data(t).feats_fine));
    z = {};
    for p = 1:7
        f = curPredVec==p;
        z{p} = f;
    end
    z = cat(3,z{:});
    pred_vecs{t} = z(:);
end
pred_vecs = cat(2,pred_vecs{:});
%
C_mean = cellfun3(@col,{training_data.feats_coarse},2);
F_mean = cellfun3(@col,{training_data.feats_fine},2);
% interaction_feats = add_feature([],C,'feats_coarse','C');
% interaction_feats = add_feature(interaction_feats,F,'feats_fine','F');
% interaction_feats = add_feature([],vl_homkermap(C_mean,1),'feats_coarse_h','C_h');
interaction_feats = add_feature([],vl_homkermap(F_mean,1),'feats_fine_h','F_h');
% interaction_feats = add_feature(interaction_feats,normalize_vec(F,1,1),'feats_fine_n','F_n1');
interaction_feats = add_feature(interaction_feats,normalize_vec(F_mean,1,2),'feats_fine_n','F_n2');
interaction_feats = add_feature(interaction_feats,...
    vl_homkermap(normalize_vec(F_mean,1,2),1),'feats_fine_n_h','F_n2_h');
%
max_vecs_fine = cat(2,max_vecs_fine{:});
interaction_feats = add_feature(interaction_feats,...
    max_vecs_fine,'max_vecs_fine','m_v_f');

interaction_feats = add_feature(interaction_feats,...
    vl_homkermap(max_vecs_fine,1),'max_vecs_fine_hkm','m_v_f_h');


% train a classifier based on: interaction features, appearance features

% first, split the detections per images.
training_per_image = {};
imgs_inds = [training_data.imgInd];
for t = 1:length(fra_db)
    if ~fra_db(t).isTrain,continue,end
    curTrainingData = training_data(imgs_inds==t);
    
    
    
    
    
end

%
% interaction_feats = add_feature(interaction_feats,...
%     single(pred_vecs),'pred_vecs','p_v');
% interaction_feats = add_feature(interaction_feats,...
%     vl_homkermap(single(pred_vecs),1),'p_v_h','p_v_h'
%%
% train_params.lambdas = .001;
train_params.toBalance = 0;
imgInds = [training_data.imgInd];
curIsTrain = [training_data.isTrain];
%curIsTrain = [fra_db(imgInds).isTrain];
%
LUT = [0 1 2 3 3 3 3 3]; % model all action objects as one
% LUT = 0:7;
% none, face, hand, 'Drink','Smoke','Blow','Brush','Phone'};
labels2 = [training_data.label];
labels2 = LUT(labels2+1);
%train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.lambdas = .001;
train_params.classNames = {'face','hand','obj'};
train_params.classes = 1:length(train_params.classNames);%[1 2 3];
train_params.minGroupSize=0;
train_params.maxGroupSize=inf;
train_params.task = 'classification';
% interaction_feats = interaction_feats([3 5]);
%
% for t = 1:length(training_data)
%     if length(training_data(t).ovp_box)~=3
%         training_data(t).ovp_box(3) = 0;
%        t
%     end
% end
% curIsTrain = [training_data.isTrain];
% values = cat(1,training_data.ovp_box);
% [v,iv] = sort(values,2,'descend');

% addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
res_interaction_feats = train_and_test_helper(interaction_feats,...
    labels2(:),...
    curIsTrain,[],train_params);
%
[~,sm_interaction] = summarizeResults_distributed(res_interaction_feats,interaction_feats,train_params);
sm_interaction
%
% sm_interaction.F_n2 = num2str(sm_interaction.F_n2);
%% demonstrate some detections
test_scores = cat(1,res_interaction_feats{6}.res_test.curScores);
test_inds = find(~curIsTrain);
test_labels = [training_data(test_inds).label];
already_seen = false(size(fra_db));
% test_labels = labels2(~test_inds);
[r,ir] = sort(test_scores(3,:),'descend');
% res_interaction_featss{1}.res_train(1).classifier_data.w
nSeenImages = 0;
for it = 1:1:length(ir)
    it                
    k = test_inds(ir(it));
    ii = training_data(k).imgInd;
    training_data(k).ovp_big_box
    if already_seen(ii),continue,end
    already_seen(ii) = true;
    %     nSeenImages = nSeenImages+1;
    %     if mod(nSeenImages,5)~=0,continue,end
    I = imdb.images_data{ii};
    clf; imagesc2(I);
    plotBoxes(training_data(k).bbox);
    %title(num2str(LUT(training_data(k).label+1)));
    dpc
end
%%
%%
% now collect hand, face, object images : for the training use the
% ground-truth, for testing use the best candidate as detected by the
% intereaction features classifier.
test_scores = cat(1,res_interaction_feats{15}.res_test.curScores);
test_inds = find(~curIsTrain);
for it = 1:length(test_inds)
    training_data(test_inds(it)).test_scores = test_scores(:,it);
end
test_inds_fra_db = [training_data(test_inds).imgInd];
face_imgs = {};
hand_imgs = {};
object_imgs = {};
face_masked = {};
hand_masked = {};
object_masked = {};
face_boxes = nan(length(fra_db),4);
obj_boxes = nan(length(fra_db),4);
hand_boxes = nan(length(fra_db),4);
% hfo_imgs = {};

% todo - what if no object was marked at training? consider it invalid
% what if it was not found at test time?
% valids_for_hands =
% valids_obj = zeros(3,length(fra_db));
valids_obj = true(3,length(fra_db));
dummy_image = ones(30,30,3,'uint8');

fillValue = [];

for t = 1:length(fra_db)
    t
    q = imgInds==t;
    curTrainingData = training_data(q);
    % if training, get the ground truth labels
    I = imdb.images_data{t};
    %     try
    f_face = [];f_hand = [];f_obj = [];
    if fra_db(t).isTrain
        curLabels = [curTrainingData.label];
        f_face = find(curLabels==1,1,'last');
        f_hand = find(curLabels==2,1,'last');
        f_obj = find(curLabels>=3,1,'last');
    else
        curScores = cat(2,curTrainingData.test_scores);
        if (~isempty(curScores))
            [~,im] = max(curScores,[],2);
            f_face = im(1);
            f_hand = im(2);
            f_obj = im(3);
        end
    end
    sz = size2(I);
    if isempty(f_face)
        valids_obj(1,t) = false;
        face_imgs{t} = dummy_image;
        face_masked{t} = dummy_image;
    else
        face_box = curTrainingData(f_face).bbox;
        face_boxes(t,:) = face_box;
        face_imgs{t} = cropper(I,face_box);
        curMask = propsToRegions(curTrainingData(f_face),sz);
        face_masked{t} = maskedPatch(I,curMask{1},true,fillValue,false);
    end
    if isempty(f_hand)
        valids_obj(1,t) = false;
        hand_imgs{t} = dummy_image;
        hand_masked{t} = dummy_image;
    else
        hand_box = curTrainingData(f_hand).bbox;
        hand_boxes(t,:) = hand_box;
        hand_imgs{t} = cropper(I,hand_box);
        curMask = propsToRegions(curTrainingData(f_hand),sz);
        hand_masked{t} = maskedPatch(I,curMask{1},true,fillValue,false);
    end
    if isempty(f_obj)
        valids_obj(1,t) = false;
        object_imgs{t} = dummy_image;
        object_masked{t} = dummy_image;
    else
        obj_box = curTrainingData(f_obj).bbox;
        obj_boxes(t,:) = obj_box;
        object_imgs{t} = cropper(I,obj_box);
        curMask = propsToRegions(curTrainingData(f_obj),sz);
        object_masked{t} = maskedPatch(I,curMask{1},true,fillValue,false);
    end
    %     if ~isTrain(t)
    %
    %         figure(1); clf;
    %         subplot(2,3,1); imagesc2(face_imgs{t});
    %         subplot(2,3,2); imagesc2(object_imgs{t});
    %         subplot(2,3,3); imagesc2(object_imgs{t});
    %
    %         subplot(2,3,4); imagesc2(face_masked{t});
    %         subplot(2,3,5); imagesc2(hand_masked{t});
    %         subplot(2,3,6); imagesc2(object_masked{t});
    %         dpc
    %     end
    %         catch e
    %             disp(e.message)
    %             valids_obj(t) = false;
    %
    %         end
end
%%
save ~/storage/misc/many_sub_images_and_boxes.mat face_boxes face_imgs face_masked hand_boxes hand_imgs hand_masked obj_boxes object_imgs object_masked
% x2(object_masked(isTrain(1:length(hand_masked)) & labels==1))
% load ~/storage/misc/many_sub_images_and_boxes.mat
%%

% nowlook at the space between the hand and the face
action_object_regions = {};
%%
for t = 1:1:length(fra_db)
    t
    %     if ~(fra_db(t).isTrain)
    I = imdb.images_data{t};
    
    curFaceBox = face_boxes(t,:);
    curHandBox = hand_boxes(t,:);
    % if there's not hand there must be an object
    if isnan(curHandBox(1)) && fra_db(t).isTrain
        curHandBox = obj_boxes(t,:);
    end
    if isnan(curFaceBox)
        if ~fra_db(t).isTrain
            faceAndHandPolygon = box2Pts(fra_db(t).I_rect);
            action_object_regions{t} = maskedPatch(I,poly2mask2(faceAndHandPolygon,size2(I)),true);
            continue
        else
            error('nan face box in training set!');
        end
    end
    
    faceAndHandPolygon=[box2Pts(curFaceBox);box2Pts(curHandBox)];
    K = convhull(faceAndHandPolygon);
    faceAndHandPolygon = poly2cw2(faceAndHandPolygon(K,:));
    
    q = imgInds==t;
    curTrainingData = training_data(q);
    curScores = cat(2,curTrainingData.test_scores);
    curBoxes =  cat(1,curTrainingData.bbox);
    
    boxPolys = boxes2Polygons(curBoxes);
    boxPolys = cellfun2(@poly2cw2,boxPolys);
    intAreas = zeros(length(boxPolys),1);
    
    
    
    %         continue
    for iPoly = 1:length(boxPolys)
        %             curIntersection = polybool2('&', boxPolys{iPoly},faceAndHandPolygon);        
        intAreas(iPoly) = polyarea2(polybool2('&', boxPolys{iPoly},faceAndHandPolygon));
    end
    aa = polyarea2(boxPolys{1});
    intAreas = intAreas/aa;
    action_object_regions{t} = maskedPatch(I,poly2mask2(faceAndHandPolygon,size2(I)),true);            
    %
    %           clf; imagesc2(I);
    % %         plotBoxes(hand_boxes(t,:),'r-','LineWidth',2);
    % %         plotBoxes(face_boxes(t,:),'g-','LineWidth',2);
    %         plotBoxes(fra_db(t).I_rect,'m--','LineWidth',2);
    %         plotPolygons(faceAndHandPolygon,'c-','LineWidth',2);
    %         dpc
    % find the best action object inside this region :-)    
    %     end
end
save ~/storage/misc/action_object_regions.mat action_object_regions

%% load  ~/storage/misc/action_object_regions.mat
face_and_hand_feats = featureExtractor.extractFeaturesMulti(action_object_regions);


%%
hand_feats = featureExtractor.extractFeaturesMulti(hand_imgs);
face_feats = featureExtractor.extractFeaturesMulti(face_imgs);
obj_feats = featureExtractor.extractFeaturesMulti(object_imgs);
hand_feats_m = featureExtractor.extractFeaturesMulti(hand_masked);
face_feats_m = featureExtractor.extractFeaturesMulti(face_masked);
obj_feats_m = featureExtractor.extractFeaturesMulti(object_masked);
save ~/storage/misc/many_sub_images_features.mat hand_feats face_feats obj_feats hand_feats_m face_feats_m obj_feats_m
% cellfun3(@size,hand_masked(1:437))
%%








%%
DD = 1000;
% DD=1
% all_feats_hof = add_feature(all_feats1,[hand_feats;face_feats;obj_feats]/DD,'hand_feats','H1');
all_feats_gfp = add_feature(all_feats1(1:2),face_and_hand_feats/DD,'proxy','proxy');
%
isTrain = [fra_db.isTrain];
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .01;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 3;
train_params.maxGroupSize = 3;
res_hof_gfp = train_and_test_helper(all_feats_gfp,labels(:),isTrain(:),valids(:),train_params);
%%
[summary,sm_hof_gfp] = summarizeResults_distributed(res_hof_gfp,all_feats_gfp,train_params);
sm_hof_gfp
%%
showSomeResults(res_hof_single,outPath,fra_db,imdb,63);
% test_labels_fra = labels(~isTrain);
%%
save ~/storage/misc/action_object_data_2.mat all_obj_data
%%
%%
isTrain = [fra_db.isTrain];
valids = true(size(isTrain));
labels = [fra_db.classID];
res_oracle = train_and_test_helper(all_feats(4),labels,isTrain,valids,train_params);
classifiers = [res_oracle{1}.res_train.classifier_data];
% plot([classifiers.w])
%%
all_obj_features = {};
all_obj_regions = {};
for t = 1:length(all_obj_data)
    t
    if ~isempty(all_obj_data{t} )
        sz = size2(imdb.images_data{t});
        I_orig = imdb.images_data{t};
        I = imResample(I_orig,[384 384],'bilinear');
        curCandidates = all_obj_data{t};
        regions = propsToRegions(curCandidates,size2(I));
        regions = cellfun2(@(x) imResample(x,sz,'nearest'),regions);
        sel_ = cellfun(@(x) any(x(:)),regions);
        regions = regions(sel_);
        curFeats = featureExtractor.extractFeaturesMulti_mask(I_orig,regions);
        all_obj_regions{t} = curCandidates(sel_);
        all_obj_features{t} = curFeats;
    end
end
%%

%%
for it = 1:length(f_test)
    t = f_test(it);
    t
    
    if ~isempty(all_obj_data{t} )
        sz = size2(imdb.images_data{t});
        I_orig = imdb.images_data{t};
        I = imResample(I_orig,[384 384],'bilinear');
        curCandidates = all_obj_data{t};
        regions = propsToRegions(curCandidates,size2(I));
        displayRegions(I,regions);
        
    end
end

f_test = find(~isTrain);
all_obj_features_t = all_obj_features(f_test);
labels_test = labels(f_test);
% res_test = apply_classifiers(res_oracle{1}.res_train,all_obj_features_t,labels,train_params);
[ws bs] = get_w_from_classifiers(res_oracle{1}.res_train);
all_scores = zeros(5,length(f_test));


img_scores = {};

for t = 1:length(all_obj_features_t)
    curFeats = all_obj_features_t{t};
    img_scores{t} = bsxfun(@plus,ws'*curFeats,bs);
    curScores = max(img_scores{t},[],2);
    all_scores(:,t) = curScores;
end
%%
% show sorted by image score.
[r,ir] = sort(all_scores,2,'descend');
class_sel = 1;
for it = 1:size(ir,2)
    k = f_test(ir(class_sel,it));
    figure(1);clf;
    
    I = imdb.images_data{k};
    
    sz = size2(I);
    I_orig = I;
    subplot(1,2,1); imagesc2(I_orig);
    I = imResample(I_orig,[384 384],'bilinear');
    curCandidates = all_obj_data{k};
    regions = propsToRegions(curCandidates,size2(I));
    regions = cellfun2(@(x) imResample(x,sz,'nearest'),regions);
    sel_ = cellfun(@(x) any(x(:)),regions);
    regions = regions(sel_);
    curImgScores = img_scores{ir(class_sel,it)};
    classScoresForImage=curImgScores(class_sel,:);
    subplot(1,2,2);
    
    
    displayRegions(I_orig,regions,classScoresForImage,'maxRegions',3);
end
%%
% make

%%
%%
%%
feats1 = feats1(1);
n = 2;
feats1(n) = all_feats(2);
%
% n = n+1;
% feats1(n) = all_feats(3);
%%
classes = [fra_db.classID];
isTrain = [fra_db.isTrain];
curValids = true(size(classes));
train_params.classes = 1:5;
train_params.lambdas = 1;
res_action_obj_context = train_and_test_helper(feats1(2:end),classes,isTrain,curValids,train_params);

%%
for t = 1:1:length(fra_db)
    t
    %if fra_db(t).classID~=1,continue,end
    if (classes(t)~=2),continue,end
    clf; subplot(1,2,1); imagesc2(getImage(conf,fra_db(t)));
    subplot(1,2,2); imagesc2(imdb.images_data{t})
    fra_db(t).classID
    dpc(.5)
end

%%
nClasses = 5;
train_params.lambdas = .001;
res_action_obj_context = train_and_test_helper(feats1(2:end),classes,isTrain,curValids,train_params);
summarizeResults(res_action_obj_context,nClasses,feats1(2:end),fra_db);
%
%%
% next: find the amount of overlap of each bounding box in the training set
% with the ground-truth bounding box.
% if this is e.g, > .5, then declare it a positive instance of this class.
% train a classifier based on fc6 features to distinguish between the true
% class labels of each class, where the candidate will be the top ranked of
% the detection action areas.
% in addition, try to concatenate the fc6 features with those of the
% action-context

%%
%%
train_inds = [obj_data.isTrain];
figure,plot(sort([obj_data(train_inds).gt_ovp]));
%%

%%

obj_context_feats = cellfun2(@(x) imResample(x,[5 5],'bilinear'),{obj_data.feats});
%%
obj_context_feats_1 = (cellfun3(@col,obj_context_feats,2));
obj_label = [obj_data.gt_ovp]>=.5;
obj_valids = true(size(obj_label));
my_feats_context = struct;
my_feats_context.name = 'object_context';
my_feats_context.feats = obj_context_feats_1;

%%
is_gt_region = [obj_data.is_gt_region];
pos_sample = train_inds & is_gt_region;
pos_sample_objs = obj_data(pos_sample);
pos_sample_inds = [pos_sample_objs.image_ind];
pos_sample_classes = [fra_db(pos_sample_inds).classID];

pos_imgs = {};
pos_context_feats = obj_context_feats_1(:,pos_sample_inds);
for t = 1:length(pos_sample_objs)
    I = imdb.images_data{pos_sample_inds(t)};
    I = imResample(I,[384 384]);
    pos_imgs{t} = cropper(I,pos_sample_objs(t).bbox);
end

pos_feats = struct;
pos_feats(1).name = 'appearance';
pos_feats(1).feats = featureExtractor.extractFeaturesMulti(pos_imgs,false);
pos_feats(2).name = 'object_context';
pos_feats(2).feats = pos_context_feats;

F_mean = cat(1,pos_feats.feats);
this_train_params = train_params;
this_train_params.classes = 1:5;
classifier_data = train_classifiers(F_mean,pos_sample_classes,[],this_train_params);


%%
train_params = struct('toBalance',0,'lambdas',.0001);
train_params.toBalance = -1;
train_params.task = 'classification';
train_params.hardnegative = false;
nClasses = 5;
train_params.classes = 1:5;
is_pos_object = [obj_data.gt_ovp]>=.5;
obj_label = [fra_db([obj_data.image_ind]).classID];
obj_label(~is_pos_object) = -1;
res_action_obj_context = train_and_test_helper(my_feats_context,obj_label,train_inds,obj_valids,train_params);
test_set = find(~[obj_data.isTrain]);
test_scores = res_action_obj_context{1}.res_test.curScores;
test_labels = obj_label(test_set);
test_ovps = [obj_data(test_set).gt_ovp];
img_inds = [obj_data(test_set).image_ind];
res_action_obj_context{1}.res_test.info
%%
%% do some visualizations
%%
uinds = unique(img_inds);
for iInd = 1:length(uinds)
    curImgInd = uinds(iInd);
    sel_ = img_inds == curImgInd;
    cur_scores = test_scores(sel_);
    I = imdb.images_data{curImgInd};
    I = imResample(I,[384 384],'bilinear');
    sel_f = find(sel_);
    boxes = cat(1,obj_data(test_set(sel_f)).bbox);
    bb = boxes; bb(:,3:4) = bb(:,3:4)-bb(:,1:2);
    bb = [bb cur_scores(:)];
    bb = bbNms(bb);
    bb(:,3:4)=bb(:,3:4)+bb(:,1:2);
    curPreds = L.preds{curImgInd};
    curScores = L.scores{curImgInd};
    bbox = bb(1,1:4);
    curScoresSoftMax = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
    showPredictions(single(cropper(I,bbox)),...
        cropper(curPreds,bbox),...
        cropper(curScores,bbox),L.labels,1);
    dpc
    %     figure(2); subplot(2,
    bb(:,3:4)=bb(:,3:4)+bb(:,1:2);
    curPreds = L.preds{curImgInd};
    curScores = L.scores{curImgInd};
    bbox = bb(1,1:4);
    curScoresSoftMax = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
    showPredictions(single(cropper(I,bbox)),...
        cropper(curPreds,bbox),...
        cropper(curScores,bbox),L.labels,1);
    dpc
    %     figure(2); subplot(2,
    bb(:,3:4)=bb(:,3:4)+bb(:,1:2);
    curPreds = L.preds{curImgInd};
    curScores = L.scores{curImgInd};
    bbox = bb(1,1:4);
    curScoresSoftMax = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
    showPredictions(single(cropper(I,bbox)),...
        cropper(curPreds,bbox),...
        cropper(curScores,bbox),L.labels,1);
    dpc
    %     figure(2); subplot(2,4,4);plotBoxes(curBox);
    %     for t = 1:min(3,size(bb,1))
    %         clf; images!du -c2(I); plotBoxes(bb(t,:));
    %         title(num2str(bb(t,end)));
    %         dpc
    %     end
    %     for it = 1:length(sel_f)
    %         t = sel_f(it);
    %         clf; imagesc2(I);
    %         plotBoxes(obj_data(test_set(t)).bbox);
    %         dpc
    %     end
end

%%
[r,ir] = sort(test_scores,'descend');
% ir = 1:length(test_scores);
already_seen = false(size(fra_db));
for it = 1:1:length(test_scores)
    it
    k = ir(it);
    curInd = test_set(k);
    cur_label = test_labels(k);
    if (cur_label==1),continue,end
    curData = obj_data(curInd);
    %x2(curData.feats(:,:,4))
    %     if curimg
    curImgInd = curData.image_ind;
    %     if curImgInd~=540,continue,end
    if already_seen(curImgInd)
        continue
    end
    already_seen(curImgInd) =true;
    %     if it < 1287,continue,end
    sum(already_seen)
    I = imdb.images_data{curImgInd};
    I = imResample(I,[384 384],'bilinear');
    curBox = curData.bbox;
    clf; imagesc2(I); plotBoxes(curBox);
    title({num2str(r(it)),num2str(test_ovps(k))});
    curPreds = L.preds{curImgInd};
    curScores = L.scores{curImgInd};
    curScoresSoftMax = bsxfun(@rdivide,exp(curScores),sum(exp(curScores),3));
    showPredictions(single(I),curPreds,curScoresSoftMax,L.labels,1);
    %     figure(2); subplot(2,4,4);plotBoxes(curBox);
    dpc
end

feats_obj = cat(2,feats_obj{:});



%%
isTrain = [fra_db.isTrain];
featureExtractor_alexnet = DeepFeatureExtractor(conf,true,16);
channelFeats = {};
for t = 1:length(fra_db)
    t
    curScores = imResample(LL.scores_coarse{t},[224 224],'bilinear');
    channelImages = {};
    for iChannel = 1:8
        channelImages{iChannel} = curScores(:,:,iChannel);
    end
    curFeats = featureExtractor_alexnet.extractFeaturesMulti(channelImages);
    for iChannel = 1:8
        channelFeats{t,iChannel} = curFeats(:,iChannel);
    end
end
all_feats_2 = [];
for t = 1:size(channelFeats,2)
    all_feats_2 = add_feature(all_feats_2,channelFeats(:,iChannel)',labels_full{t},labels_full{t});
end

train_params = struct('classes',1:5,'toBalance',0,'lambdas',.001);
train_params.toBalance =0;
train_params.task = 'classification';
train_params.hardnegative = false;
nClasses = 5;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.minGroupSize = 7;
train_params.maxGroupSize = 8;

labels = [fra_db.classID];
isTrain = [fra_db.isTrain];
res_2 = train_and_test_helper(all_feats_2,labels(:),isTrain,[],train_params);

[summary,sm_coarse_and_fine] = summarizeResults_distributed(res_2,all_feats_2,train_params);
sm_coarse_and_fine



addpath('/home/amirro/code/3rdparty/vedaldi_detection');

%%
for t = 1:30:length(fra_db)
    t
    if fra_db(t).isTrain,continue,end
    showPredsHelper2(fra_db,imdb,t);
    dpc; continue
    %t=1
    V = LL.scores_fine{t};
    %     M = max(V(:,:,3:end),[],3);
    %     [V,IV] = max(V(:,:,3:end),[],3);
    %
    %     x2(IV)
    %     I = imdb.images_data{t};
    %     x2(I);
    %
    x2(V(:,:,2))
    x2(M)
    dpc
    
end


addpath('/home/amirro/code/3rdparty/matconvnet-fcn-master/utils/')
%% go over some feature maps to find detection ability of probs
outPath = '~/storage/fra_action_fcn';
    labels_full = {'none','face','hand','drink','smoke','bubbles','brush','phone'};    
f_test = find(~isTrain);
for t = 3:11:length(f_test)
    k = f_test(t);
%     if isTrain(k),continue,end
    p = j2m(outPath,fra_db(k));
    L = load(p);
    coarse_probs = L.scores_full_image;
    coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
    fine_probs = L.scores_hires;
    full_hires_path = '~/storage/fra_action_fcn_hires';
    L_full = load(j2m(full_hires_path,fra_db(k)));
%     fine_probs = L_full.scores_hires_full;
%     fine_probs = bsxfun(@rdivide,exp(fine_probs),sum(exp(fine_probs),3));
    [~,coarse_pred] = max(coarse_probs,[],3);
    [~,fine_pred] = max(fine_probs,[],3);
    %     zoomBox = inflatebbox(region2Box(imdb.labels{k}>2),4,'both');
    % zoomBox = inflatebbox(fra_db(k).faceBox,2.5,'both');
    %     showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1,zoomBox);
    %     showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2,zoomBox);
    I = im2single(imdb.images_data{k});
    V = fine_probs+coarse_probs;
    S = max(V(:,:,4:end),[],3);
    [subs,vals] = nonMaxSupr( double(S), 5, .1, 5);
    
    clf; subplot(1,2,1); imagesc2(I);
    subplot(1,2,2); imagesc2(sc(cat(3,S,I),'prob'));
    plotPolygons(fliplr(subs),'r+','LineWidth',5);
    showCoords(bsxfun(@minus,fliplr(subs),[15 0]));
    dpc
            
%     [h1_coarse,h2_coarse] = showPredictions(single(imdb.images_data{k}),coarse_pred,coarse_probs,labels_full,1);
%     [h1_fine,h2_fine] = showPredictions(single(imdb.images_data{k}),fine_pred,fine_probs,labels_full,2);
    
end

%%

%%
DD = 1000;
% DD=1
all_feats1_t = all_feats1([1 2]);
all_feats_object_proxy = add_feature(all_feats1_t,face_and_hand_feats/DD,'PROXY','PROXY');
% all_feats_hof = add_feature(all_feats_hof,hand_feats_m/DD,'hand_feats_m','HM1');
% all_feats_hof = add_feature(all_feats_hof,face_feats_m/DD,'face_feats_m','FM1');
all_feats_hof = add_feature(all_feats_hof,obj_feats_m/DD,'obj_feats_m','OM1');
%
% all_feats_object_proxy = all_feats1;
isTrain = [fra_db.isTrain];
valids = true(size(isTrain));
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .0001;
train_params.classNames = {'Drink','Smoke','Blow','Brush','Phone'};
train_params.standardize=false;
train_params.classes = 1:5;
train_params.minGroupSize = 1;
train_params.maxGroupSize= 3;
res_proxy = train_and_test_helper(all_feats_object_proxy,labels(:),isTrain(:),valids(:),train_params);
%%
%%
% res_coarse_and_fine = res;
[summary,sm_proxy] = summarizeResults_distributed(res_proxy,all_feats_object_proxy,train_params);
sm_proxy
% x2(face_imgs(1:50:end));
% save summary.mat summary sm_hof