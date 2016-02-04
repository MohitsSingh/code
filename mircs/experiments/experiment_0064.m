%% Experiment 0062 - 29/4/2015
%% baseline model, show how and when it fails.% 
% 1. Extract fc6 features from stanford 40:
% images, person bounding boxes, faces, upper bodies
% 2. compute SVM to classify them
initpath;
config;
networkPath = 'imagenet-vgg-s.mat';
load ~/storage/mircs_18_11_2014/s40_fra_faces_d.mat;
s40_fra = s40_fra_faces_d;
load ~/storage/misc/all_dnn_feats.mat;
all_dnn_feats_deep = all_dnn_feats;
% load ~/storage/misc/all_dnn_feats_faces.mat
load ~/storage/misc/all_dnn_feats_head.mat
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
addpath('/home/amirro/code/3rdparty/myqueue_1.1/')
% all_dnn_feats_deep;
%%
% apply nin to s40 images for classification....
addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11'));
ninet = load('/home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/data/cifar-nin/net-epoch-44.mat');
% load /home/amirro/code/3rdparty/matconvnet-1.0-beta11/examples/data/nin_normalization
net = ninet.net;
net.layers = net.layers(1:end-1);
I1 = normalize_data_cifar(I,cifar_normalization);
res = vl_simplenn(net, I1, [], [], 'disableDropout', true);
s40_tiny = {};
s40_tiny_people = {};
%%
ticID = ticStatus('shrinking s40',.5,.5,true);
for t = 2318:length(s40_fra)
%     t
    [I,I_rect] = getImage(conf,s40_fra(t));    
    s40_tiny{t} = imResample(I,[32 32],'bilinear',true);
    I_person = cropper(I,I_rect);
    s40_tiny_people{t} = imResample(I_person,[32 32],'bilinear',true);
    tocStatus(ticID,t/length(s40_fra));    
end
% tocStatus(ticID,1);
save ~/storage/misc/s40_tiny.mat s40_tiny s40_tiny_people
%%
% extract nin features for these images
isTrain = [s40_fra.isTrain];
s40_tiny_n = normalize_data_cifar(s40_tiny,cifar_normalization);
s40_tiny_people_n = normalize_data_cifar(s40_tiny_people,cifar_normalization);

batches = batchify(length(s40_tiny),100);
%[data,inds] = splitToRanges(1:9532,100);


feats = {};
feats_people = {};
for iBatch = 1:length(batches)
    disp(100*iBatch/length(batches))
    curBatch = batches{iBatch};
    x = vl_simplenn(net, s40_tiny_n(:,:,:,curBatch), [], [], 'disableDropout', true);    
    feats{iBatch} = reshape((x(end-2).x),[],length(curBatch));
    x = vl_simplenn(net, s40_tiny_people_n(:,:,:,curBatch), [], [], 'disableDropout', true);    
    feats_people{iBatch} = reshape((x(end-2).x),[],length(curBatch));
end

save ~/storage/misc/s40_tiny_feats.mat feats feats_people

feats = cat(2,feats{:});
feats_people = cat(2,feats_people{:});
%%
train_params.classes = conf.class_enum.SMOKING;
isTrain = [s40_fra.isTrain];
% lambdas_old = lambdas;
res_train_tiny = train_classifiers(feats_people(:,isTrain),train_labels,train_params,toBalance,.00001);
res_test = apply_classifiers(res_train_tiny,feats_people(:,~isTrain),test_labels,train_params);
res_test.info
figure,plot(res_test.curScores)
% clf; showSorted(s40_tiny_people(~isTrain),-res_test.curScores);
%%
% whosbetter ~/storage/misc/s40_tiny_feats.mat
res = vl_simplenn(net, s40_tiny_n(:,:,:,1:100), [], [], 'disableDropout', true);

% size(ninet.net.layers{end-4}
% for t = 24:-1:1
%     clc
%     t
%     ninet.net.layers{t}
%     pause
% end
% 
% % we want layer 19 in ninet
% 
% (size(ninet.net.layers{19}.weights{1}))
%%


all_dnn_feats = struct;
valids = true(size(s40_fra));
for u = 1:length(all_dnn_feats_deep)
    u
    all_dnn_feats(u).imageID = s40_fra(u).imageID;
    all_dnn_feats(u).feats_crop = cat(1,all_dnn_feats_deep(u).feats_crop(1).x); % should be crop
    all_dnn_feats(u).feats_crop_tiled = cat(1,col(all_dnn_feats_deep(u).feats_crop_tiled.x));
    all_dnn_feats(u).feats_full = cat(1,all_dnn_feats_deep(u).feats_full(1).x);
    all_dnn_feats(u).feats_full_tiled = cat(1,col(all_dnn_feats_deep(u).feats_full_tiled.x));
    all_dnn_feats(u).feats_crop_deep = cat(1,all_dnn_feats_deep(u).feats_crop_deep(1).x); % should be crop
    all_dnn_feats(u).feats_crop_deep_tiled = cat(1,col(all_dnn_feats_deep(u).feats_crop_deep_tiled.x));
    all_dnn_feats(u).feats_full_deep = cat(1,all_dnn_feats_deep(u).feats_full_deep(1).x);
    all_dnn_feats(u).feats_full_deep_tiled = cat(1,col(all_dnn_feats_deep(u).feats_full_deep_tiled.x));
    curHeadFeats = all_dnn_feats_head(u).result;
    if (isempty(curHeadFeats))
        valids(u) = false;
        continue;
    end
    for iType = 1:length(curHeadFeats)
        all_dnn_feats(u).(curHeadFeats(iType).type) = curHeadFeats(iType).feat;
    end
end
%%

extents = fieldnames(all_dnn_feats(1));
extents = setdiff(extents,{'imageID'});
% all_dnn_feats = rmfield(all_dnn_feats,'feats_deep');
[train_ids,~,train_labels] = getImageSet(conf,'train');
[test_ids,~,test_labels] = getImageSet(conf,'test');
% feat_layers = [16 16 18 18];
% feat_extent = {'feats_full','feats_crop','feats_full','feats_crop'};
% some training  parameters...
train_params.features.normalize_all = false;
train_params.features.normalize_each = false;
train_params.classes = [conf.class_enum.DRINKING];
% extract features from several image regions...
train_features = getImageFeatures(all_dnn_feats,train_ids);
% train_features is a cell array with several feature types.
%% test
test_features = getImageFeatures(all_dnn_feats,test_ids);

%% try all non-empty feature subsets (ablation study).

%%
% cool, now try to predict the usefullness of each features on its own.
% nTotalClasses = 40;
% classes = [conf.class_enum.DRINKING];%,conf.class_enum.SMOKING,conf.class_enum.BLOWING_BUBBLES,conf.class_enum.BRUSHING_TEETH];
% classes = my_classes
classes = 1:40;
nTotalClasses = length(classes);
avg_prec_est = zeros(nTotalClasses,length(train_features));
lambdas =  [1e-5 1e-6 1e-7];% 1e-6]
train_valids = valids(1:length(train_ids));
toBalance = 0;

%%
% now, train each class using several options:
% 1. each subset independently.
% 1. best classes by order (ir 1:9)
extents
warning('using only  features of entire image...');
test_results = struct('target_class',{},'feature_subset',{},'performance',{},'classifier_data',{});
for iClass = 1:nTotalClasses
    nExp = 0;
    train_params.classes = classes(iClass);
    % 1: single subsets
    %for iSubset = 1:length(train_features)
    
    for iSubset = 18
        fprintf('class: %d, subset: %d\n',train_params.classes, iSubset);
        nExp = nExp+1;
        feature_subset = iSubset;
        test_results(iClass,nExp) = train_with_subset_and_test(iClass,train_features,feature_subset,...
            train_params,train_labels,test_labels,valids,train_ids,test_features,lambdas);
    end
end
%
conf.class_enum
conf.classes(9)

%%
%
for t = 1:length(I_test)
    I = I_test{t};
    detections = occludeAndExtract(I,param,w_g);
    z = visualizeTerm(detections(:,5),boxCenters(detections),size2(I));
    clf; subplot(2,1,1);
    imagesc2(I);
    subplot(2,1,2); imagesc2(sc(cat(3,exp(-z),I),'prob'))
    max(detections(:,5))
    pause    
end
%%

net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
% net_deep = init_nn_network('imagenet-vgg-s.mat');

for t = 1:length(net_deep.layers)
    curLayer = net_deep.layers{t};
    if (isfield(curLayer,'weights'))
        curLayer.filters = curLayer.weights{1};
        curLayer.biases = curLayer.weights{2};
        curLayer = rmfield(curLayer,'weights');
        net_deep.layers{t} = curLayer;
    end
end
net_deep.layers = net_deep.layers(1:32);
layers = 33;
% net_deep.layers = net_deep.layers(1:15);
%%

% net = load('imagenet-vgg-f.mat');
%%
T = s40_fra_faces_d(~[s40_fra_faces_d.isTrain]);
%%

% save ~/storage/misc/classifier_data_fc6.mat test_results
% get all of the classifier results and find most confused classes.
%%
%all_scores = zeros(nTotalClasses,length(test_ids));
dd = [test_results.performance];

infos = [dd.info];
% mean([infos.ap])


[aa,iaa] = sort([infos.ap],'descend')

conf.classes(iaa)

c_data = [test_results.classifier_data];
ww = [c_data.w];
% save ~/storage/misc/ww.mat ww
all_scores = cat(1,dd.curScores);
label_matrix = zeros(size(all_scores));
for u = 1:nTotalClasses
    label_matrix(u,:) = test_labels == u;
end

%[z,iz] = sort(all_scores,2,'descend');
[q,iq] = max(all_scores,[],1);

imagesc(confMatrix(test_labels,iq,nTotalClasses))
[q,idx_pred] = max(all_scores);
CM = confMatrix( test_labels, idx_pred, nTotalClasses );
imagesc(CM);
counts = sum(CM,2);
cm1 = bsxfun(@rdivide,CM,counts);
imagesc(cm1); colormap jet
wrong_labels = idx_pred~=test_labels';
% classifier_ws = 
[t,it] = sort(q,'descend');
%%

addpath('/home/amirro/code/3rdparty/myqueue_1.1');

%for ik = length(t):-300:1
for ik = 1:length(t)
    k = it(ik);
%     k = ik
%     if (wrong_labels(k))
        curScore = t(ik);
        curClass = test_labels(k);
        pred_class = idx_pred(k);
        pred_class_str = conf.classes{pred_class};
        w = ww(:,pred_class);
        curID = test_ids{k};
        featsPath = j2m('~/storage/s40_kriz_fc6_block_5_2',curID);
        if (~exist(featsPath,'file')),continue,end
        % show only false alarms
        %[I,R] = getRegionImportance(conf,curID, w, net_deep,[13]);
                        
        [I,R] = getRegionImportance(conf,curID, w, net_deep,[32]);                        
%         zzz = makeTiles(I,25,2);
        curName = sprintf('%3.3f, gt: %s,pred: %s',curScore,conf.classes{curClass},pred_class_str);
        clf; imagesc2(R);
         h = title(curName); 
        set(h,'interpreter','none');        
        dpc;
%     end
end



%%
I = getImage(conf,curID);
I = imResample(I,2,'bilinear');
% x2(I)
% I = ;
z = vl_simplenn(net_deep,single(I)-128);
m = reshape(reshape(z(16).x,[],4096)*w(1:end-1),size2(z(16).x))
% x2(m)
% x2(I)

%%
UU = {};
for t = 1:length(s40_fra)
    t
    featsPath = j2m('~/storage/s40_kriz_fc6_block_5_2',s40_fra(t));
    UU{t} = load(featsPath);
    UU{t}.imageID = s40_fra(t).imageID;
end
UU = [UU{:}];
%%
%%
iClass = 26;
iSubset = 17;

feats0 = train_features(iSubset).feats;
T = train_ids;

isTrain = [s40_fra.isTrain];
tr0 = false(size(T));
tr0(1:2:end) = true;
z_0(tr0) = true;

tr1 = ~tr0;
train_params.classes = iClass;
% train classifier using global features

res_train_0 = train_classifiers(feats0(:,tr0),train_labels(tr0),train_params,toBalance,lambdas);
res_train_1 = train_classifiers(feats0(:,tr1),train_labels(tr1),train_params,toBalance,lambdas);

% get significant regions contributing to classification (or errors...)
w0 = res_train_0.classifier_data_p.w;
w1 = res_train_1.classifier_data_p.w;
important_regions_0 = getImportantRegions(conf,w1,train_ids(tr0),UU);
important_regions_1 = getImportantRegions(conf,w0,train_ids(tr1),UU);

important_regions = {};
for t = 1:length(important_regions_0)
    important_regions{2*(t-1)+1} = important_regions_0(t);
end
for t = 1:length(important_regions_1)
    important_regions{2*t} = important_regions_1(t);
end
important_regions = [important_regions{:}];

% focus only on features which were in highly scoring in the first place
scores0 = zeros(1,size(feats_local,2));
scores0(tr1) = w0(1:end-1)'*feats0(:,tr1);
scores0(tr0) = w1(1:end-1)'*feats0(:,tr0);

[u,iu] = sort(scores0,'descend');
labels_u = train_labels(iu)==iClass;
f_plus = find(labels_u==1,100,'first')
f_minus = find(labels_u==0,100,'first')
sel_pos = iu(f_plus);
sel_neg = iu(f_minus);
min_score = u(f_minus(end));

pos_feats = [important_regions(sel_pos).feats_outside];
neg_feats = [important_regions(sel_neg).feats_outside];
[x_g,y_g] = featsToLabels(pos_feats,neg_feats);
window_classifier = Pegasos(x_g,y_g,...
        'lambda', .001,'forceCrossVal',true);

% concatenate with global regions, and train a new classifier
% feats_local = zeros(size(feats0,1),length(train_ids));
% feats_local(:,tr0) = cat(2,important_regions_0.feats_inside);
% feats_local(:,tr1) = cat(2,important_regions_1.feats_inside);


% [z,iz] = sort(scores0,'descend');
% tr1_ids = train_ids(tr1);
% getImportantRegions(conf,w0,tr1_ids(iz(1:10:end)),UU,true);

% baseline classifier
res_train = train_classifiers(feats0,train_labels,train_params,toBalance,lambdas);
feats_test = test_features(iSubset).feats;
test_results_baseline = apply_classifiers(res_train,feats_test,test_labels,train_params);
% baseline + window features
res_train_w = train_classifiers([feats0;feats_local],train_labels,train_params,toBalance,lambdas);
important_regions_test = getImportantRegions(conf,res_train.classifier_data_p.w,test_ids,UU);

for_windows = test_results_baseline.curScores > min_score;
% for_windows = 1:length(test_labels);
rescoring = window_classifier.w(1:end-1)'*[important_regions_test(for_windows).feats_outside];

% % important_regions_test = getImportantRegions(conf,res_train.classifier_data_p.w,test_ids(for_windows),UU,true);


%%
newScores = test_results_baseline.curScores;
newScores(for_windows) = newScores(for_windows)+0+.1*rescoring;
vl_pr(2*(test_labels==iClass)-1,newScores)
%%
% feats_test_w = [feats_test;cat(2,important_regions_test.feats_inside)];
% test_results_w = apply_classifiers(res_train_w,feats_test_w,test_labels,train_params);


disp(['baseline: ' num2str(test_results_baseline.info.auc)])
% disp(['windowed: ' num2str(test_results_w.info.auc)])

%%
% nnz(wrong_labels)/length(wrong_labels)
%for t = 1:length
figure(1)
nToShowPerClass = inf;%15;
nWindows = 5^2;
imgHeight = 256;
startWithFalse = false;
showOnlyFalse = false
showOnlyTrue = true
zz = 0;
dryRun = false;
neededIDS = {};

for iClass = conf.class_enum.BRUSHING_TEETH 
% for iClass = 1:length(conf.classes)
    % show sorted results....
    a = test_results(iClass);
    a = a.performance.info.ap
    %a = a.info
    %a = a.ap
    %test_results(iClass).performance.info.ap
    curScores = test_results(iClass).performance.curScores;
    w = test_results(iClass).classifier_data.w;
    %vl_pr(2*(test_labels==iClass)-1,curScores)
    [r,ir] = sort(curScores,'descend');
    n = length(r);
%     ir = 1:n
    sorted_ids = test_ids(ir);
    sorted_labels = test_labels(ir)==iClass;
    %
    seenFalse = false    
    u = 100*cumsum(sorted_labels)/sum(sorted_labels);
    v = 0;    
    v_pos = 0;
    v_neg = 0;
    q = 0;
    for t = 1:n
        if (showOnlyTrue && ~sorted_labels(t))
            continue
        end
        if showOnlyFalse && sorted_labels(t)
            continue
        end
        if startWithFalse
            if ~sorted_labels(t)
                seenFalse = true;
            else
                if ~seenFalse
                    continue
                end
            end
        end
        v = v+1;    
        curID = sorted_ids{t};
        
        if (sorted_labels(t))
            v_pos = v_pos+1;
            if (v_pos > nToShowPerClass)
                continue
            end
        else
            v_neg = v_neg+1;
            if (v_neg > nToShowPerClass)
                continue
            end
        end
        zz=zz+1;
        q = q+1;
        fprintf('trgt class:%s, %d out of %d, (got %%%.2f positives) (%s)\n',conf.classes{iClass},...
            t,n,u(t),curID);        
        if (dryRun)
            neededIDS{end+1} = curID;
            continue
        end                        
        [I,I_rect] = getImage(conf,curID);        
        [I,R,rects] = getRegionImportance(conf,curID,w,net_deep,32);
        clf; imagesc2(R)
        dpc;
    end
end
