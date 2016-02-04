if (~exist('initialized','var'))
    cd ~/code/mircs
    initpath;
    config;
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
%     addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');
%     rmpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/'));
    matConvNetPath = '/home/amirro/code/3rdparty/matconvnet-fcn-master/matconvnet/';
    addpath(matConvNetPath);
    addpath(fullfile(matConvNetPath,'examples'));
    addpath(fullfile(matConvNetPath,'matlab'));
    
    
    load('~/storage/misc/pascal_action_imdb.mat');
    
    initialized = true;
    %     addpath('~/code/3rdparty/matconvnet-1.0-beta16/matlab');
    %     featureExtractor = DeepFeatureExtractor(conf,true,17);
    gpuDevice(2)
    featureExtractor16 = DeepFeatureExtractor(conf,true,35);
    featureExtractor19 = DeepFeatureExtractor(conf,true,41,'/home/amirro/storage/matconv_data/imagenet-vgg-verydeep-19.mat');
    load ~/storage/misc/pascal_action_imdb.mat
    addpath('~/code/mircs/cvpr2016/');
    addpath('~/code/mircs/learning');
    addpath('/home/amirro/code/3rdparty/matconvnet-fcn-master/utils');
    %     L = load('~/storage/misc/action_object_pred_full_2_w_classes.mat');
    load ~/storage/misc/action_names
    %      = row(action_names);
    train_params.classNames = row(VOCopts.actions);
    train_params = struct('classes',1:length(train_params.classNames),'toBalance',0,'lambdas',.001);
    
    train_params.classes = [1 3:11];
    train_params.classNames = row(VOCopts.actions(2:end));
    
    %     train_params.classes = [1 3:11];
    train_params.task = 'classification';
    train_params.minGroupSize = 0;
    train_params.maxGroupSize = inf;
    train_params.hardnegative = false;
    train_params.standardize=false;
    train_params.minGroupSize=1;
    train_params.maxGroupSize=inf;
    %       train_params.classNames = VOCopts.actions;
    
    %     {'phoning',...
    %     'playinginstrument',...
    %     'reading',...
    %     'ridingbike',...
    %     'ridinghorse',...
    %     'running',...
    %     'takingphoto',...
    %     'usingcomputer',...
    %     'walking'};
    outPath = '~/storage/pascal_action_fcn_full';
    imdb = pascal_imdb;
    isTrain = imdb.isTrain;
    %     nClasses = 9;
end
%%
% 1->phoning
train_params.classNames = row(VOCopts.actions(2:end));
train_params.classes = [1 3:11];
%%
for ZZ = 1:11
    class_sel = train_params.classes(ZZ);
    class_name = train_params.classNames(ZZ)
    acc = 0;
    for t = 1:length(images_data)
        %         t
        z = images_data(t);
        if z.label_vec(class_sel)==1
            I = imdb.images_data{z.imgInd};
            clf; imagesc2(I);
            plotBoxes(z.bbox);
            dpc
            acc = acc+1;
            if acc==5
                break
            end
        end
    end
end
%%

% baseline...

% for each image, obtain:
% 1. global image descriptor
% 2. local (person bounding box) image descriptor
% 3.

% order the images so they are consistent with the pascal training and
% testing.
images_data = struct('imgInd',{},'imageID',{},'bbox',{},'bbox_id',{},'global_desc',{},'local_descs',{},'isTrain',{},'label_vec',{});
n = 0;
for t = 1:length(imdb.imageIds)
    imgInd = t;
    imageID = imdb.imageIds{t};
    curBoxes = imdb.img_boxes{t};
    %curLabels = reshape(imdb.img_label_vecs{t},9,[]);
    curLabels = imdb.img_label_vecs{t};
    curBoxIds = imdb.img_box_ids{t};
    for iBox = 1:size(curBoxes,1)
        n = n+1;
        images_data(n).imageID = imageID;
        images_data(n).imgInd = imgInd;
        images_data(n).label_vec = curLabels(:,iBox);
        images_data(n).bbox_id = curBoxIds(iBox);
        images_data(n).bbox = curBoxes(iBox,:);
        images_data(n).isTrain = imdb.isTrain(t);
    end
end
%%

localPatches = {};
for t = 1:length(images_data)
    t
    curImg = imdb.images_data{images_data(t).imgInd};
    curBox = round(images_data(t).bbox);
    %     curBox = round(makeSquare(curBox,true));
    localPatches{t} = cropper(curImg,curBox);
end
%
%%
% extrac the multiscale features for each image and each of the local
% patches
featsPath = '~/storage/pascal_vgg_ms_feats';
ensuredir(featsPath);

% for t = 1:length(imdb.images_data)
%     img_id = imdb.imageIds{t};
%     fName = fullfile(featsPath,[imd_id '_16.mat']);
% end
%imshow(imrotate(localPatches{t}(:,:,1),-90))
% gpuDevice(2)
globalFeats16 = {};
globalFeats19 = {};
%%GLOBALS
ppp = randperm(length(imdb.images_data));
%%
for ik = 1:length(imdb.images_data)
    if (mod(ik,50)==0)
        disp(ik)
    end
    ii = ppp(ik);
 %     '16'
    img_id = imdb.imageIds{ii};
    fName = fullfile(featsPath,[img_id '_globals16.mat']);
    if ~exist(fName,'file')
        x = featureExtractor16.my_extract_dnn_feats_multiscale(imdb.images_data(ii));
        save(fName,'x');
        ik
    else
                load(fName)
                assert(none(isnan(x)))
            globalFeats16{ii} = x;
    end
    
    %     continue
%     '19'
    fName = fullfile(featsPath,[img_id '_globals19.mat']);
    if ~exist(fName,'file')
        x = featureExtractor19.my_extract_dnn_feats_multiscale(imdb.images_data(ii));
        save(fName,'x');
        ik
    else
                load(fName);
                globalFeats19{ii} = x;
            assert(none(isnan(x)))
    end
    
    
end

%
localFeats16 = {}
localFeats19 = {}
%%PATCHES
ppp = randperm(length(localPatches));

for ik = 1:length(localPatches)
    if (mod(ik,50)==0)
        disp(ik)
    end
    ii = ppp(ik);
%     ii
    %     img_id = imdb.imageIds{image};
    patch_id = ['patch_' num2str(ii)];
    fName = fullfile(featsPath,[patch_id '_16.mat']);
    if ~exist(fName,'file')
        x = featureExtractor16.my_extract_dnn_feats_multiscale(localPatches(ii));
        save(fName,'x');
        ik
    else
                load(fName)
                    assert(none(isnan(x)))
            localFeats16{ii} = x;
        
    end
    
    fName = fullfile(featsPath,[patch_id '_19.mat']);
    if ~exist(fName,'file')
        ik
        x = featureExtractor19.my_extract_dnn_feats_multiscale(localPatches(ii));
        save(fName,'x');
    else
        load(fName);
        localFeats19{ii} = x;
        assert(none(isnan(x)))
%         
    end
    
end
%%
% localFeats19 = loadlFeats19;
globalFeats16 = cat(2,globalFeats16{:});
globalFeats19 = cat(2,globalFeats19{:});
localFeats16 = cat(2,localFeats16{:});
localFeats19 = cat(2,localFeats19{:});
%%
for t = 1:length(images_data)
    images_data(t).global_desc16 = globalFeats16(:,images_data(t).imgInd);
    images_data(t).global_desc19 = globalFeats19(:,images_data(t).imgInd);
    %     images_data(t).global_desc19_simple = globalFeats19_simple(:,images_data(t).imgInd);
    %     images_data(t).global_desc = images_data(t).global_desc.*single(images_data(t).global_desc>0);
    images_data(t).local_desc16 = localFeats16(:,t);
    images_data(t).local_desc19 = localFeats19(:,t);
    %     images_data(t).local_desc19_simple = localFeats19_simple(:,t);
    %     images_data(t).local_desc = images_data(t).local_desc.*single(images_data(t).local_desc>0);
end
% eats19_simple = featureExtractor19.my_extract_dnn_feats_simple(imdb.images_data);

% localFeats19_orig = featureExtractor19.extractFeaturesMulti(localPatches);
% localFeats19_orig-localFeats19_simple
% globalFeats_fc6 = featureExtractor.extractFeaturesMulti(imdb.images_data);

labels = cat(2,images_data.label_vec)';
valids = labels(:,2)==0;
valids = [];
% [~,labels] = max(labels,[],1);
isTrain = imdb.isTrain([images_data.imgInd])>0;
%%
imagesc(labels)
baseLinePath = '~/storage/misc/baseline_pascal';
if exist(baseLinePath,'file')
    load(baseLinePath);
else
    
    
    % changed the globalfeats16....
    %     imgs_local =
    %     feats_local = featureExtractor.extractFeaturesMulti(imgs_local,false);
    %globalFeats = featureExtractor.extractFeaturesMulti(imdb.images_data,false);
    feats_global16 = cat(2,images_data.global_desc16);
    feats_local16 = cat(2,images_data.local_desc16);
    feats_global19 = cat(2,images_data.global_desc19);
    feats_local19 = cat(2,images_data.local_desc19);
    
    %     use_simple_feats=true
    %
    %     if (use_simple_feats)
    %         feats_global19 = cat(2,images_data.global_desc19_simple);
    %         feats_local19 = cat(2,images_data.local_desc19_simple);
    %     end
    
    to_relu=true;
    to_normalize = false;
    to_normalize_all = true;
    if (to_relu)
        feats_global16 = feats_global16.*(feats_global16>0);
        feats_local16 = feats_local16.*(feats_local16>0);
        feats_global19 = feats_global19.*(feats_global19>0);
        feats_local19 = feats_local19.*(feats_local19>0);
    end
    
    if to_normalize
        feats_global16 = normalize_vec(feats_global16);
        feats_local16 = normalize_vec(feats_local16);
        feats_global19 = normalize_vec(feats_global19);
        feats_local19 = normalize_vec(feats_local19);
    end
%     FFF  =[feats_local16+feats_local19+feats_global16+feats_global19];
    FFF  =[feats_local16;feats_local19;feats_global16;feats_global19];
%         FFF  =[feats_local19;feats_global19];
%     FFF  =[feats_local16;feats_global16]+[feats_local19;feats_global19];
    if (to_normalize_all)
        FFF = normalize_vec(FFF);
    end
    
    %     labels = [fra_db.classID];
    %     isTrain = [fra_db.isTrain]
    %        D =500;
    %all_feats = add_feature([],([feats_global16;feats_local16]),'G','G');
    %     FFF = vl_homkermap(FFF,1);
    all_feats = add_feature([],FFF,'G');
    
    %      all_feats = add_feature([],normalize_vec(feats_local+feats_local,2),'L_and_G','L_and_G');
    %
    %     all_feats = add_feature(all_feats,feats_local,'L','L');
    %
    %     addpath('~/code/3rdparty/liblinear-1.95/matlab/')
    %
    fffff = [1 3:11];
    train_params.classNames = row(train_params.classNames);
    %     train_params.classes_orig = train_params.classes
    train_params.classes  = fffff(1:2);
    train_params.task = 'classification';
    train_params.toBalance = 0;
    %%
    train_params.lambdas = .001;
    train_params.minGroupSize=1;
    %     train_params.classes = 5;
    %     train_params.classNames = 'playinginstrument';
    res = train_and_test_helper(all_feats,labels,isTrain(:),[],train_params);
    [~,res_sm] = summarizeResults_distributed(res,all_feats,train_params);
    res_sm
    % show what you've learned
    %%
    test_inds = find(~isTrain);
    test_scores = res{1}.res_test(1).curScores;
    [r,ir] = sort(test_scores,'descend');
    for t = 1:length(test_scores)
        
        t
        % %         figure,plot(res{1}.res_test(1).recall,res{1}.res_test(1).precision)
        
        k = test_inds(ir(t));
        if images_data(k).label_vec(1)==1,continue,end
        curImg = imdb.images_data{images_data(k).imgInd};
        curBox = round(images_data(k).bbox);
        
        clf; imagesc2(curImg); plotBoxes(curBox);
        %         images_data(k).label_vec
        dpc
    end
    
    
    %%
    % res = apply_classifiers(res,feats_global(:,~sel_train),labels(~sel_train),train_params);
    save(baseLinePath,'res','all_feats','train_params','objects_boxed','objects_masked','objects_masked_tight',...
        'imgs_local');
end


%% fine, add  the interaction features.
outPath = '~/storage/pascal_action_fcn_results/';

coarse_sum_features = {};
coase_max_features = {};
fine_sum_features = {};
fine_max_features = {};

LL.scores_coarse = {};
LL.scores_fine = {};


for t = 1:length(imdb.images)
    t
    %     p = j2m(outPath,fra_db(t));
    %     L = load(p);
    masked_images = {};
    %     coarse_probs = LL.scores_coarse{t};
    %fine_probs = LL.scores_fine{t};
    
    p = fullfile(outPath,[imdb.imageIds{t} '.mat']);
    L = load(p);
    fine_probs = L.scores_hires;
    fine_probs = bsxfun(@rdivide,exp(fine_probs),sum(exp(fine_probs),3));
    
    coarse_probs = L.scores_full_image;
    coarse_probs = bsxfun(@rdivide,exp(coarse_probs),sum(exp(coarse_probs),3));
    
    LL.scores_coarse{t} = coarse_probs;
    LL.scores_fine{t} = fine_probs;
    
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
save ~/storage/misc/action_probs_pascal.mat coarse_sum_features coase_max_features fine_sum_features fine_max_features
%%
coarse_sum_features = cat(2,coarse_sum_features{:});
coase_max_features =  cat(2,coase_max_features{:});
fine_sum_features =  cat(2,fine_sum_features{:});
fine_max_features =  cat(2,fine_max_features{:});


for t = 1:length(images_data)
    q=  images_data(t).imgInd;
    images_data(t).coarse_sum_features = coarse_sum_features(:,q);
    images_data(t).coase_max_features = coase_max_features(:,q);
    images_data(t).fine_sum_features = fine_sum_features(:,q);
    images_data(t).fine_max_features = fine_max_features(:,q);
    %     images_data(t).local_desc19_simple = localFeats19_simple(:,t);
    %     images_data(t).local_desc = images_data(t).local_desc.*single(images_data(t).local_desc>0);
end
%%
C1m = cat(2,images_data.coase_max_features);
F1m = cat(2,images_data.fine_max_features);
C1s = cat(2,images_data.coarse_sum_features);
F1s = normalize_vec(cat(2,images_data.fine_sum_features));
my_feats = add_feature([],FFF,'G');
my_feats = add_feature(my_feats,vl_homkermap(C1m,1),'coarse_max_chi2','cm');
% all_feats1 = add_feature(all_feats1,vl_homkermap(fine_sum_features,1),'fine_sum');
my_feats = add_feature(my_feats,vl_homkermap(F1m,1),'fine_max_chi2','fm');
% my_feats = add_feature(my_feats,vl_homkermap((F1m+C1m)/2,1),'fine_max_chi2','fmcm');

% my_feats = add_feature(my_feats,vl_homkermap(C1s,1),'fine_max_chi2','cs');
% my_feats = add_feature(my_feats,vl_homkermap(F1s,1),'fine_max_chi2','fs');
% all_feats1 = add_feature(all_feats1,vl_homkermap(fi
train_params.classes_orig = [1 3:11];

train_params.classes  = train_params.classes_orig(1:2);
train_params.task = 'classification';
train_params.toBalance = 0;
train_params.lambdas = .00001;
train_params.minGroupSize=3;
% train_params.classes = [1 3:11];

%     train_params.classes = 5;
%     train_params.classNames = 'playinginstrument';
res = train_and_test_helper(my_feats,labels,isTrain(:),[],train_params);
[~,res_sm] = summarizeResults_distributed(res,my_feats,train_params);
res_sm
%%

% select different classes and see some results...
test_inds = find(~isTrain);
test_scores = res

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
all_feats1 = add_feature(all_feats1,vl_homkermap(fine_max_features+coase_max_features,N),'fine_max_chi2','fine_and_coarse');
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
%% extract a few candidates from each image
candidateParams = struct('useRawScores',false,...
    'useLocalMaxima',false,...
    'usePrediction',true,...
    'useMaxObj',true);
training_data = getAllObjectCandidates(LL, candidateParams , imdb, fra_db);
% save ~/storage/misc/hand_obj_face_training_data_with_regions.mat training_data

%%


training_data_cell = {};
%%

candidateParams = struct('useRawScores',false,...
    'useLocalMaxima',true,...
    'usePrediction',false,...
    'useMaxObj',false);

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

save ~/storage/misc/hand_obj_face_training_data_with_regions_localmax.mat training_data_cell

%save ~/storage/misc/hand_obj_face_training_data_with_regions_pred_and_max.mat training_data_cell
%%
% load ~/storage/misc/hand_obj_face_training_data_with_regions_pred_and_max.mat
% training_data = cat(2,training_data_cell{:});
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
    curTrainingData(boxArea < 100) = [];
    lenAfter = length(curTrainingData);
    fprintf('%d(%d)\n',lenAfter,lenBefore);
    curBoxes = cat(1,curTrainingData.bbox_orig);
    curBoxesBig = cat(1,curTrainingData.bbox);
    % remove boxes which are far away from both hands and faces.
    localScores = LL.scores_fine{t};
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
    [patches_orig,context_feats_orig] = extractContextFeatures(I,localScores,curBoxes);
    [patches_big,context_feats_big] = extractContextFeatures(I,localScores,curBoxesBig);
    curImgInds = ones(size(curTrainingData))*t;
    newData = curTrainingData;
    for u = 1:length(newData)
        newData(u).patch_orig = patches_orig{u};
        newData(u).patch_big = patches_big{u};
        newData(u).ovp_orig_box = ovps_orig(u,:);
        newData(u).ovp_big_box = ovps_big(u,:);
        newData(u).context_feats_orig = context_feats_orig{u};
        newData(u).context_feats_big = context_feats_big{u};
    end
    newData_cells{t} = newData;
end
%%
save ~/storage/misc/newData_cells.mat newData_cells -v7.3
%%
%%




% remove all boxes outside of person rectangles
sel_ = {};
for t = 1:length(newData_cells)
    curData = newData_cells{t};
    curRect = imdb.rects(t,:);
    curBoxes = cat(1,curData.bbox_orig);
    [ovp,int] = boxesOverlap(curBoxes,curRect);
    [~,~,a] = BoxSize(curBoxes);
    sel_{t} = int>.7*a;
end

sel_ = cat(1,sel_{:});

patches = cellfun3(@(x) {x.patch_orig},newData_cells,2);
context_feats = cellfun3(@(x) {x.context_feats_big},newData_cells,2);
context_feats = cellfun3(@col,context_feats,2);
cur_train_inds = cellfun3(@(x) [x.isTrain],newData_cells,2);
all_ovps = cellfun3(@(x) cat(1,x.ovp_orig_box),newData_cells,1);
all_boxes = cellfun3(@(x) cat(1,x.bbox_orig),newData_cells,1);
all_img_inds = cellfun3(@(x) cat(1,x.imgInd),newData_cells,1);

patches = patches(sel_);
context_feats = context_feats(:,sel_);
cur_train_inds = cur_train_inds(sel_);
all_ovps = all_ovps(sel_,:);
all_boxes = all_boxes(sel_,:);
all_img_inds = all_img_inds(sel_);


cur_train_params = train_params;
cur_train_params.classes = [0 1];
cur_train_params.classNames = {'none','object'};
sel_ = 3;
sel_pos = all_ovps(:,sel_) > .5;
sel_neg = all_ovps(:,sel_) <= .2;

curLabels = zeros(size(all_ovps,1),1);
curLabels(sel_pos)=1;
curLabels(sel_neg)=0;

%%
patch_appearance = featureExtractor16.extractFeaturesMulti(patches);
save ~/storage/misc/patch_appearance.mat patch_appearance


%%
cur_train_params.lambda = .01;
interaction_feats_new = add_feature([],context_feats,'context_feats','context_feats');
interaction_feats_new = add_feature(interaction_feats_new,patch_appearance,'p_app','p_app');

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
for t = 1:length(fra_db)
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
face_and_hand_feats = featureExtractor16.extractFeaturesMulti(action_object_regions);
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

%%
hand_feats = featureExtractor16.extractFeaturesMulti(hand_imgs);
face_feats = featureExtractor16.extractFeaturesMulti(face_imgs);
obj_feats = featureExtractor16.extractFeaturesMulti(object_imgs);
hand_feats_m = featureExtractor16.extractFeaturesMulti(hand_masked);
face_feats_m = featureExtractor16.extractFeaturesMulti(face_masked);
obj_feats_m = featureExtractor16.extractFeaturesMulti(object_masked);
save ~/storage/misc/many_sub_images_features.mat hand_feats face_feats obj_feats hand_feats_m face_feats_m obj_feats_m
% cellfun3(@size,hand_masked(1:437))
%%
DD = 1000;
% DD=1
% all_feats_hof = add_feature(all_feats1,[hand_feats;face_feats;obj_feats]/DD,'hand_feats','H1');
all_feats_hof = add_feature(all_feats1,face_and_hand_feats/DD,'proxy','proxy');
%all_feats_hof = add_feature(all_feats_hof,face_feats/DD,'face_feats','F1');
%all_feats_hof = add_feature(all_feats_hof,obj_feats/DD,'obj_feats','O1');
%all_feats_hof = add_feature(all_feats_hof,obj_feats_m/DD,'obj_feats_m','O1_m');
%all_feats_hof = add_feature(all_feats_hof,hand_feats/DD,'hand_feats','H1');

% all_feats_hof = add_feature(all_feats_hof,hand_feats_m/DD,'hand_feats_m','HM1');
% all_feats_hof = add_feature(all_feats_hof,face_feats_m/DD,'face_feats_m','FM1');
% all_feats_hof = add_feature(all_feats_hof,obj_feats_m/DD,'obj_feats_m','OM1');
%
isTrain = [fra_db.isTrain];
% valids = min(valids_obj,[],1);
labels = [fra_db.classID];
train_params.lambdas = .001;
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
save summary_single.mat sm_hof_single
% end
%%
train_params.lambdas = .001;
train_params.minGroupSize =4;
train_params.maxGroupSize= inf;
res_hof_one_out = train_and_test_helper(all_feats_hof,labels(:),isTrain(:),valids(:),train_params);

train_params.lambdas = .001;
train_params.minGroupSize =4;
res_hof_one_out1 = train_and_test_helper(all_feats_hof,labels(:),isTrain(:),valids(:),train_params);
[summary,sm_hof_one_out1] = summarizeResults_distributed(res_hof_one_out1,all_feats_hof,train_params);

%%
% res_coarse_and_fine = res;
[summary,sm_hof_one_out] = summarizeResults_distributed(sm_hof_one_out1,all_feats_hof,train_params);
save summary_one_out.mat sm_hof_one_out

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
        curFeats = featureExtractor16.extractFeaturesMulti_mask(I_orig,regions);
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
pos_feats(1).feats = featureExtractor16.extractFeaturesMulti(pos_imgs,false);
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
    %     figure(2); subplot(2,4,4);plotBoxes(curBox);
    %     for t = 1:min(3,size(bb,1))
    %         clf; imagesc2(I); plotBoxes(bb(t,:));
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
VOCopts.testset = 'test';
[ids] = textread(sprintf(VOCopts.imgsetpath,VOCopts.testset),'%s');

imagePaths = cellfun2(@(x) sprintf(VOCopts.imgpath,x),ids);
exist(imagePaths{100},'file')

