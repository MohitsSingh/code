% 6/5/2015
%---------
% Action Object Detection - a structured output framework for action object detection
% and description.
cd ~/code/mircs
initpath;
config;
%addpath('/home/amirro/code/3rdparty/edgeBoxes/');
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');

% rmpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta12/'));
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');

% vl_compilenn('enableGpu',true);

%%
addpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained');
install
%%
load fra_db;
%bbLabeler({'hand'},myTmpDir,myTmpDir);
% bbLabeler({'hand'},conf.imgDir,'/home/amirro/data/Stanford40/annotations/hands/');
% update hands!
annoPath = '/home/amirro/storage/data/Stanford40/annotations/hands';
annoPath_orig = '/home/amirro/data/Stanford40/annotations/hands/';
addpath('/home/amirro/code/3rdparty/gaimc');
% bbLabeler({'hand'},conf.imgDir,annoPath)
listOfNeededPaths = {};
for t = 1:length(fra_db)
    imgData = fra_db(t);
    bbPath = j2m(annoPath,imgData,'.jpg.txt');
    bb = [];
    obj = bbGt('bbLoad',bbPath);
    if (~isempty(obj))
        bb = cat(1,obj.bb);
        bb = bb(:,1:4);
        bb(:,3:4) = bb(:,3:4)+bb(:,1:2);
    end
    fra_db(t).hands = bb;
end
% save fra_db fra_db
%     end
%%
params = defaultPipelineParams();
%% TRAINING



% 1. Define a graph structure
nodes = struct('name',{},'type',{},'spec',{},'params',{},'bbox',{},'poly',{},'valid',{});
edges = struct('v',{},'connection',{});
isTrain = [fra_db.isTrain];
posClass = 4; % brushing teeth
isClass = [fra_db.classID] == posClass;
isValid = true(size(fra_db));%[fra_db.isValid];
% findImageIndex(fra_db,'brushing_teeth_064.jpg')
train_pos = isClass & isTrain & isValid;
train_neg = ~isClass & isTrain & isValid;
f_train_pos = find(train_pos);
f_train_neg = find(train_neg);
test_pos = isClass & ~isTrain & isValid;
test_neg = ~isClass & ~isTrain & isValid;
f_test_pos = find(test_pos);
f_test_neg = find(test_neg);
f_train = find(isTrain & isValid);
f_test = find(~isTrain & isValid);
% check the coverage of the ground-truth regions using edgeboxes.
% clf;
% 1. define graph-structure
% 2. define a way to transform graph to image
% 3. extract features from graph
% 4. optimize cost function over graph.
%% Define a graph, whos nodes are:
% mouth, object, hand
% interaction between node is important
% the state of a node may by not only location, but also orientation
nodes(1).name = 'mouth';
nodes(1).type = 'region';
nodes(1).spec.size = .25;
nodes(2).name = 'obj';
nodes(2).type = 'poly';
nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
nodes(3).name = 'hand';
nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
nodes(3).type = 'region';
%state of a node may by not only location, but also orientation
nodes(1).name = 'mouth';
nodes(1).type = 'region';
nodes(1).spec.size = .25;
nodes(2).name = 'obj';
nodes(2).type = 'poly';
nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
nodes(3).name = 'hand';
nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
nodes(3).type = 'region';

edges(1).v = [1 2];
edges(1).connection = 'anchor';
edges(2).v = [2 3];
edges(2).connection = 'anchor';
for iNode = 1:length(nodes)
    nodes(iNode).valid = true;
end
% sample good configurations, configurations are assignments to the graph,
% and each configuration is defined a set of polygons defining image
% regionsate of a node may by not only location, but also orientation
nodes(1).name = 'mouth';
nodes(1).type = 'region';
nodes(1).spec.size = .25;
nodes(2).name = 'obj';
nodes(2).type = 'poly';
nodes(2).spec = struct('avgLength',1,'avgWidth',.3);
nodes(3).name = 'hand';
nodes(3).spec = struct('avgLength',.7,'avgWidth',.7);
nodes(3).type = 'region';

%%
%% let's do this in several stages, with differing levels of complexity.
% The regions can be: bounding boxes or oriented polygons
% The geometric relation between the nodes can be constrained or
% non-constrained
% The node samples can be either any region or regions derived from some
% proposal method (e.g) edge boxes
configurations = {};
params.gt_mode = 'box_from_poly';
params.rotate_windows = true;
params.cand_mode = 'polygons';
params.cand_mode = 'boxes';
params.cand_mode = 'segments';
params.feature_extraction_mode = 'bbox';
params.holistic_features = false; %TODO - remeber to check this as a baseline...
params.interaction_features = 1;
% imgData = fra_db(1);
% x2(getImage(conf,imgData))
% plotBoxes(imgData.faceBox)
regionSampler = RegionSampler();
regionSampler.clearRoi();
nSamples = 20;
theta_start = 0;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
lengths = .5;
widths = .5;
params.sampling.thetas = theta_start:b:theta_end;%0:10:350;
params.sampling.lengths = lengths;
params.sampling.widths = widths;
params.sampling.nBoxThetas = 8;
%params.sampling.boxSize = [.5 .7 1];
params.sampling.boxSize = [.5 .7 1];
params.sampling.maxThetaDiff = 50;
%[.3 .5 .7];

%%
featureExtractor = DeepFeatureExtractor(conf,true);
%%
cur_set = f_train_pos;
toVisualize = false;
theta_start = 0;
%theta_end = theta_start+180;
nSamples = 15;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
% predict the amount of ovp...
roi_pos_patches = {};
roi_neg_patches = {};
params_coarse = params;
params_coarse.sampling.boxSize = 1;

%% stage -3 : train a classifier over mouth area alone: is this a mouth with the object of
% interest touching it or not?
pos_mouths = {};
neg_mouths = {};
for t = 1:length(fra_db)
    t
    if ~isTrain(t),continue,end
    if ~isValid(t),continue,end
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    h = faceBox(3)-faceBox(1);
    mouthBox = round(inflatebbox(imgData.mouth,h/2,'both',true));
    %faceBox = inflatebbox(faceBox,1.5,'both',false);
    mouthImage = cropper(I,mouthBox);
    if isClass(t)
        pos_mouths{end+1} = mouthImage;
    else
        neg_mouths{end+1} = mouthImage;
    end
end

getBad = @(x)cellfun3(@isempty,x)
pos_mouths(getBad(pos_mouths)) = [];
neg_mouths(getBad(neg_mouths)) = [];
mouth_pos_feats = featureExtractor.extractFeaturesMulti(pos_mouths);fprintf('\n');
mouth_neg_feats = featureExtractor.extractFeaturesMulti(neg_mouths);fprintf('\n');
[w_mouth b_mouth] = concat_and_learn(gather(mouth_pos_feats),gather(mouth_neg_feats));



%% show some stage -3 results
%
cur_set = f_test;
% set_labels = {}
% set_scores = {}
set_images = {}

for it = 1:length(cur_set)
    if mod(it,30)==0
        it
    end
    t = cur_set(it);
    imgData = fra_db(t);
    %     imgData.imageID
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    h = faceBox(3)-faceBox(1);
    mouthBox = round(inflatebbox(imgData.mouth,h/2,'both',true));
    mouthImage = cropper(I,mouthBox);
    
    set_images{it} = mouthImage;
    continue
    clf;
    subplot(1,2,1); imagesc2(I); plotBoxes(mouthBox);
    subplot(1,2,2); imagesc2(mouthImage);
    r = featureExtractor.extractFeaturesMulti(mouthImage);
    curScore = w_mouth'*r+b_mouth;
    title(num2str(curScore));
    dpc
end
goods = cellfun3(@(x) ~isempty(x),set_images);
set_feats = featureExtractor.extractFeaturesMulti(set_images(goods));
set_scores = w_mouth'*gather(set_feats)+b_mouth;
set_labels = isClass(cur_set(goods))*2-1;
vl_pr(set_labels,set_scores)
showSorted(set_images(goods),set_scores)
%%

% show some stage -1 results.
% cur_set = f_test_neg;%f_test_neg;
cur_set = f_test_pos;
% cur_set = f_test;%f_test_neg;
% cur_classes = f_train
% % % all_scores = zeros(size(cur_set));
theta_start = 0;
nSamples = 25;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
% cur_set = cur_set(randperm(length(cur_set)));
%it = showOrder;
% cur_set = cur_set(fliplr(showOrder));
params_coarse.sampling.boxSize = 1
for it = 1:10%length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    params_coarse.sampling.nBoxThetas = 10;
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    rois = sampleAround(gt_graph{1},inf,h,params_coarse,I,false);
    rois = arrayfun3(@(x) x.bbox, rois,1);
    roiPatches = multiCrop2(I,round(rois));
    curFeats = featureExtractor.extractFeaturesMulti(roiPatches);
    curScores = gather(w_int'*curFeats+b_int);
    % % %     all_scores(it) = max(curScores);
    % % %     continue
    [r,ir] = sort(curScores,'descend');
    %S = computeHeatMap_regions(I,roiMasks,normalise(curScores).^2,'max');
    S = computeHeatMap(I,[rois ,normalise(curScores(:)).^2],'max');
    
    figure(1); clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2);
    RR = sc(cat(3,S,im2double(I)),'prob');
    imagesc2(RR);
    title(num2str(max(curScores)));
    
    dpc
    
end

%sz = cellfun3(@size2,neg_mouths)
%% stage -2 : train a coarse detector over entire face bounding box.
neg_faces = {};pos_faces = {};
for t = 1:length(fra_db)
    t
    if ~isTrain(t),continue,end
    if ~isValid(t),continue,end
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    faceBox = inflatebbox(faceBox,1.5,'both',false);
    faceImage = cropper(I,round(faceBox));
    if isClass(t)
        pos_faces{end+1} = faceImage;
    else
        neg_faces{end+1} = faceImage;
    end
end

%%
face_pos_feats = featureExtractor.extractFeaturesMulti(pos_faces);
face_neg_feats = featureExtractor.extractFeaturesMulti(neg_faces);
[w_face b_face] = concat_and_learn(face_pos_feats,face_neg_feats);

%% show some stage -2 results

cur_set = f_test_neg;
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    faceBox = inflatebbox(faceBox,1.5,'both',false);
    faceImage = cropper(I,round(faceBox));
    clf; imagesc2(I); plotBoxes(faceBox);
    faceFeats = featureExtractor.extractFeaturesMulti(faceImage);
    curScore = w_face'*faceFeats+b_face;
    title(num2str(curScore));
    dpc
end
%%
%% test faces...
test_faces = {};
for it = 1:length(f_test)
    t = f_test(it)
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    faceBox = inflatebbox(faceBox,1.5,'both',false);
    faceImage = cropper(I,round(faceBox));
    test_faces{end+1} = faceImage;
end
test_face_feats = featureExtractor.extractFeaturesMulti(test_faces);
showSorted(test_faces,w_face'*test_face_feats);

%% stage -1 training: train region-of-interaction detector
cur_set = f_train_pos;
roi_pos_patches = {};
for it = 1:length(cur_set)
    it
    %     profile on
    t = cur_set(it);
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    params_coarse.cand_mode = 'boxes';
    rois = sampleAround(gt_graph{1},inf,h,params_coarse,I,false);
    rois = arrayfun3(@(x) x.bbox, rois,1);
    g = poly2mask2(cellfun2(@(x) x.poly,gt_graph),size2(I));
    roiMasks = box2Region(rois,size2(I));
    [~,ints,uns] = regionsOverlap(roiMasks,g);
    ints = ints./cellfun(@nnz,roiMasks)';
    roiPatches = multiCrop2(I,round(rois));
    roi_pos_patches{end+1} = roiPatches(ints > .2);
    roi_neg_patches{end+1} = roiPatches(ints < .05);
    %     profile viewer
end
% z = getLogPolarMask(50,10,2);
%%
roi_pos_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_pos_patches{:}));
roi_neg_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_neg_patches{:}));
%%
[w_int b_int] = concat_and_learn(gather(roi_pos_feats),gather(roi_neg_feats));
%%
%% show some stage -1 results.
% cur_set = f_test_neg;%f_test_neg;
cur_set = f_test_neg;
% cur_set = f_test;%f_test_neg;
% cur_classes = f_train
% % % all_scores = zeros(size(cur_set));
theta_start = 0;
nSamples = 25;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
% cur_set = cur_set(randperm(length(cur_set)));
%it = showOrder;
% cur_set = cur_set(fliplr(showOrder));
params_coarse.sampling.boxSize = 1;
params_coarse.sampling.nBoxThetas = 25;
for it = 1:length(cur_set)
    it
    %     tic
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    params_coarse.nodes = nodes;
    [rois,curScores] = scoreCoarseRois(conf,imgData,params_coarse,featureExtractor,w_int,b_int);
    
    [r,ir] = sort(curScores,'descend');
    %S = computeHeatMap_regions(I,roiMasks,normalise(curScores).^2,'max');
    S = computeHeatMap(I,[rois ,normalise(curScores(:)).^2],'max');
    %      toc
    %     S = computeHeatMap_regions(I,roiMasks(ir(1)),normalise(curScores(ir(1))),'max');
    figure(1); clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2);
    RR = sc(cat(3,S,im2double(I)),'prob');
    imagesc2(RR);
    title(num2str(max(curScores)));
    
    %saveas(gcf,['/home/amirro/notes/images/2015_06_15/coarse_' imgData.imageID '.png'])
    
    dpc
    %     [~,ints,uns] = regionsOverlap(roiMasks,g);
    %     ints = ints./cellfun(@nnz,roiMasks)';
    %     roi_pos_patches{end+1} = roiPatches(ints > .5);
    %     roi_neg_patches{end+1} = roiPatches(ints < .1);
end

%%
test_labels = 2*isClass(f_test)-1;
%vl_pr(test_labels,all_scores)
[sortedScores,showOrder] = sort(all_scores,'descend');
%delay=0,trues=[],displayTrue=0,indexto
displayImageSeries(conf,fra_db(f_test(showOrder)),0,[],0,sortedScores)


%% show some images based on face scores only
face_scores = w_face'*test_face_feats;
[sortedScoresFace,showOrderFace] = sort(face_scores,'descend');
displayImageSeries(conf,fra_db(f_test(showOrderFace)),0,[],0,sortedScoresFace)

%f_test
face_scores_all = zeros(size(fra_db));
face_scores_all(f_test) = face_scores;
%vl_pr(test_labels,face_scores)
% all_scores_bkp = all_scores;
%showSorted(fra_db(f_test),all_scores)
%% learn each of the parts independently
%% stage 1: object graphs  :
cur_set = f_train_pos;
params.cand_mode = 'boxes';
all_pos_feats = struct;
debugging = false;
params.interaction_features = true;
true_configs = {};
params.nodes = nodes;
params.nParts = 2;
params.debugging = false;
params.restrictAngle = false;
params_box = params;
trainingSets = struct('sel_pos',f_train_pos(1:50),'sel_neg',f_train_neg(1:50));
coarse_data = struct('w',w_int,'b',b_int,'params',params_coarse);
% [rois,curScores] = scoreCoarseRois(conf,imgData,coarse_data.params,featureExtractor,coarse_data.w,coarse_data.b);

params_box.nParts = 2;
params_box.features.partFeats = true;
features_box = collectTrainingSamples(conf,fra_db,trainingSets,params_box,featureExtractor,[]);
models_box = train_model(features_box,params_box,params_box.nParts);
trainingSets = struct('sel_pos',f_train_pos(1:1),'sel_neg',f_train_neg(1:50));
params_seg = params;
params_seg.cand_mode = 'segments';
params_seg.restrictAngle = true;
features_box = collectTrainingSamples(conf,fra_db,trainingSets,params_seg,featureExtractor,models_box,coarse_data);

cur_set = f_test_pos;
params_box_debug = params_box;
params_box_debug.debugging = true;
params_box_debug.restrictAngle = true;
results_box = applyLearnedModel(conf,fra_db,cur_set,params_box_debug,featureExtractor,models_box,coarse_data);
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    gt_graph = gt_graph(1:2);
    if (any(cellfun(@(x) ~isfield(x,'roiMask') || nnz(x.roiMask)==0 ,gt_graph)))
        warning('skipping due to missing element...')
        continue
    end
    %     continue
    %displayRegions(I,cellfun2(@(x) x.roiMask,gt_graph))
    min_gt_ovp = .5;
    if any(sum(cellfun3(@(x) x.bbox,gt_graph,1),2)==0)
        disp('skipping current example since some nodes are missing');
        continue
    end
    gt_configurations = sample_configurations(imgData,I,min_gt_ovp,gt_graph,regionSampler,params);
    
    if debugging
        visualizeConfigurations(I,gt_configurations);
        continue
    end
    [curPartFeats,curIntFeats,curShapeFeats] = configurationToFeats2(I,gt_configurations,featureExtractor,params);
    all_pos_feats(it).partFeats = curPartFeats;
    all_pos_feats(it).intFeats = curIntFeats;
    all_pos_feats(it).shapeFeats = curShapeFeats;
    %all_pos_feats{end+1} =
end
%
% negative samples
cur_set = f_train_neg;
debugging = false;
all_neg_feats = struct;
params.conf = conf;
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    min_gt_ovp = 0;
    params.nSamples = 5;
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    gt_graph = gt_graph(1:2);
    negParams = params;
    negParams.cand_mode = 'boxes';
    configs = sample_configurations(imgData,I,0,gt_graph,regionSampler,negParams);
    if debugging
        visualizeConfigurations(I,configs);
        %         dpc
    end
    [curPartFeats,curIntFeats,curShapeFeats] = configurationToFeats2(I,configs,featureExtractor,negParams);
    all_neg_feats(it).partFeats = curPartFeats;
    all_neg_feats(it).intFeats = curIntFeats;
    all_neg_feats(it).shapeFeats = curShapeFeats;
end
%% train a classifier
% save ~/storage/data/tmp.mat all_pos_feats all_neg_feats
save ~/storage/data/w_int.mat w_int b_int

[w_config b_config] = concat_and_learn(all_pos_feats,all_neg_feats);
nParts = 3;
[w_parts b_parts] = learn_parts(all_pos_feats,all_neg_feats,nParts)

% nParts = 3;
% [models_parts,models_links] = learn_parts_2(all_pos_feats,all_neg_feats,nParts);
nParts = 3;
[models_parts_box,models_links_box,models_shapes_box] = learn_parts_2(all_pos_feats,all_neg_feats,nParts);


% % [w_light b_light info_light] = concat_and_learn(all_pos_feats_light,all_neg_feats_light);
%%
cur_set = f_test_pos;
% cur_set = f_test_neg;
% cur_set = cur_set(randperm(length(cur_set)));
close all
% cur_set = f_test;
debugging = true;

for it = 1:length(cur_set)
    t = cur_set(it);
    imgData = fra_db(t);
    %     if ~any(strfind(imgData.imageID,'phon')),continue,end
    I = getImage(conf,imgData);
    [I_sub,mouthBox,candidates] = getCandidateRegions(conf,imgData);
    displayRegions(I_sub,candidates.masks);
    %
    min_gt_ovp = 0;
    %[configs,scores] = findBestConfiguration(imgData,I,regionSampler,params);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    regions = getCandidateRegions(conf,imgData);
end
%% testing
cur_set = f_test_pos;
% cur_set = f_test_neg;
% cur_set = cur_set(randperm(length(cur_set)));
close all
% cur_set = f_test;
debugging = true;
if ~debugging
    all_results_with_int = struct('iImage',{},'coarseBox',{},'coarseScore',{},'bestConfig',{},'bestConfigScore',{},...
        'class',{},'imageID',{});
    all_results_no_int = struct('iImage',{},'coarseBox',{},'coarseScore',{},'bestConfig',{},'bestConfigScore',{},...
        'class',{},'imageID',{});
end
% cur_set = f_test(showOrder1);

for it = 4:length(cur_set)
    t = cur_set(it);
    imgData = fra_db(t);
    if ~any(strfind(imgData.imageID,'phon')),continue,end
    I = getImage(conf,imgData);
    
    %     [I_sub,mouthBox,candidates] = getCandidateRegions(conf,imgData);
    %     displayRegions(I_sub,candidates.masks);
    min_gt_ovp = 0;
    %[configs,scores] = findBestConfiguration(imgData,I,regionSampler,params);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    regions = getCandidateRegions(conf,imgData);
    % constrain search using the coarse method!
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    params_coarse.cand_mode = 'boxes';
    params_coarse.sampling.nBoxThetas = 15;
    rois_coarse = sampleAround(gt_graph{1},inf,h,params_coarse,I,false);
    boxes_coarse = arrayfun3(@(x) x.bbox, rois_coarse,1);
    roiPatches = multiCrop2(I,round(boxes_coarse));
    curFeats = featureExtractor.extractFeaturesMulti(roiPatches);
    coarseScores = w_int'*curFeats+b_int;
    [r,ir] = sort(coarseScores,'descend');
    S = computeHeatMap(I,[boxes_coarse ,normalise(coarseScores(:)).^2],'max');
    %     S = computeHeatMap_regions(I,roiMasks(ir(1)),normalise(curScores(ir(1))),'max');
    figure(1); clf;
    vl_tightsubplot(1,2,1); imagesc2(I);
    vl_tightsubplot(1,2,2);
    RR = sc(cat(3,S,im2double(I)),'prob');
    imagesc2(RR);
    title(num2str(max(coarseScores)));
    %     dpc;continue
    [m,im] = max(coarseScores);
    startTheta = rois_coarse(im).theta;
    params.nSamples = inf;
    params.sampling.boxSize = [.2 .3 .5 .7];
    % coarse to fine - find consistent region
    curRoi = rois_coarse(ir(1));
    curSampling = params.sampling;
    curSampling.thetas = 0:5:360;
    curSampling.widths = .3;
    curSampling.lengths = 1;
    rr = gt_graph{1};
    rr.theta = curRoi.theta;
    rr.endPoint = boxCenters(rr.bbox);
    curRois = getCandidateRoisPolygons2( rr ,...
        h,curSampling,true);
    roiMasks = arrayfun2(@(x) poly2mask2(x.xy,size2(I)), curRois);
    curFeats = featureExtractor.extractFeaturesMulti_mask(I,roiMasks);
    scores2= w_int'*curFeats+b_int;
    figure(2)
    displayRegions(I,roiMasks,scores2,0,3);
    continue
    %S = computeHeatMap(I,[boxes_coarse ,normalise(coarseScores(:)).^2],'max');
    
    
    %displayRegions(I,roiMasks);
    
    % % %     load(j2m('~/storage/fra_db_seg',imgData)); cands = cadidates; clear cadidates;
    % % %     params.sampling.lengths = [.2 .3 .5 .7];
    % % %     params.sampling.widths = [.2 .3];
    % % %     seg_boxes = cands.bboxes;
    % % %     nBoxes = size(seg_boxes,1);
    % % %     masks = cands2masks(cands.cand_labels, cands.f_lp, cands.f_ms);
    % % %     sum_on_x = squeeze(sum(masks,1));
    % % %     sum_on_y = squeeze(sum(masks,2));
    % % %     my_seg_boxes = zeros(length(masks),4);
    % % %
    % % %     for t = 1:size(masks,3)
    % % %         xmin = find(sum_on_x(:,t),1,'first');
    % % %         xmax = find(sum_on_x(:,t),1,'last');
    % % %         ymin = find(sum_on_y(:,t),1,'first');
    % % %         ymax = find(sum_on_y(:,t),1,'last');
    % % %         my_seg_boxes(t,:) = [xmin ymin xmax ymax];
    % % %     end
    
    % % %     masks = row(squeeze(mat2cell2(masks,[1,1,size(masks,3)])));
    % % %     cands.masks = masks;
    % % %     cands.bboxes = my_seg_boxes;
    % % %     [configs,scores] = findBestConfiguration3(imgData,I,gt_graph,params,featureExtractor,models_parts,models_links,startTheta,cands);
    
    mouthCenter = imgData.mouth;
    x2(I); plotPolygons(mouthCenter,'r+')
    faceBox = imgData.faceBox;
    h = faceBox(3)-faceBox(1);
    mouthBox = round(inflatebbox(mouthCenter,h,'both',true));
    x2(I); plotBoxes(mouthBox);
    I_sub = cropper(I,mouthBox);
    %masks_sub = cropper(masks,mouthBox);
    
    curCands = im2mcg(I_sub,'fast',true);
    masks_sub = curCands.masks;
    
    %     masks_sub = cellfun3(@(x) cropper(x,mouthBox),masks,3);
    nonZeros = squeeze(sum(sum(masks_sub,1),2));
    goods = nonZeros > 0 & nonZeros < prod(size2(I_sub));
    masks_sub1 = row(squeeze(mat2cell2(masks_sub,[1,1,size(masks_sub,3)])));
    masks_sub1 = removeDuplicateRegions(masks_sub1);
    %     displayRegions(I_sub,masks_sub1);
    sal_opts.show = false;
    maxImageSize = 50;
    sal_opts.maxImageSize = maxImageSize;
    sal_opts.useSP = true;
    sal_opts.pixNumInSP = round(prod(size2(I_sub)/5)^.5)
    [sal1,sal_bd,resizeRatio,sp_data] = extractSaliencyMap(im2uint8(I_sub),sal_opts);
    
    roiAreas = cellfun3(@nnz,masks_sub1);
    sal1_ = imResample(sal1,size2(masks_sub1{1}));
    roiSaliency = cellfun3(@(x) sum(sal1_(x)),masks_sub1);
    
    M = addBorder(zeros(size2(I_sub)),1,1)>0;
    touchesBorder = cellfun3(@(x) any(x(M(:))), masks_sub1);
    [regions,ovp,sel_] = chooseRegion(I_sub,masks_sub1,.5);
    displayRegions(I_sub,regions,ovp);
    %q = displayRegions(I_sub,masks_sub1,touchesBorder.*roiSaliency./roiAreas,0,1);
    
    XY = cellfun2(@bwperim,masks_sub1);
    
    
    q = displayRegions(I_sub,masks_sub1,touchesBorder,0)
    
    z = zeros(size2(M));
    z(end/2,end/2) = 1; z = bwdist(z);
    dist_to_center = cellfun3(@(x) [min(z(x(:))) max(z(x(:)))],masks_sub1);
    
    myScores = (dist_to_center(:,1) < 5) + (dist_to_center(:,2) > 59);
    myScores(myScores<2) = -inf;
    
    displayRegions(I_sub,masks_sub1,myScores);
    %x2(sal1_>.1)
    
    %     I = imResample(I,.25);
    %gt_graph = gt_graph(1:end-1);
    % % %     params.interaction_features = false;
    % % %     [configs,scores] = findBestConfiguration(imgData,I,gt_graph,regionSampler,params,featureExtractor,models_parts,models_links,startTheta)
    % % %     [mm,imm] = max(scores);
    % % %     if ~debugging
    % % %         all_results_no_int(t).iImage = t;
    % % %         all_results_no_int(t).imageID = imgData.imageID;
    % % %         all_results_no_int(t).class = imgData.class;
    % % %         all_results_no_int(t).coarseBox = boxes_coarse(im,:);
    % % %         all_results_no_int(t).coarseScore = m;
    % % %         all_results_no_int(t).bestConfig = configs(imm);
    % % %         all_results_no_int(t).bestConfigScore = mm;
    % % %     end
    % % %     params.interaction_features = true;
    %x2( cropper(cands.superpixels,faceBox))
    %     params.interaction_features = false;
    [mm,imm] = max(scores);
    if ~debugging
        all_results_with_int(t).iImage = t;
        all_results_with_int(t).imageID = imgData.imageID;
        all_results_with_int(t).class = imgData.class;
        all_results_with_int(t).coarseBox = boxes_coarse(im,:);
        all_results_with_int(t).coarseScore = m;
        all_results_with_int(t).bestConfig = configs(imm);
        all_results_with_int(t).bestConfigScore = mm;
    end
    
    if debugging
        %configs = sample_configurations(imgData,I,0,gt_graph,regionSampler,params);
        % % % %     curFeats = configurationToFeats2(I,configs,featureExtractor,params);
        % % % %     curScores = w_config'*cat(2,curFeats{:});
        curMasks=visualizeConfigurations(I,configs,scores,5,0,boxes_coarse(im,:),...
            [sprintf('%03.0f_',it),imgData.imageID]);
    end
end
%%
% show some results...
res_classes = 2*([fra_db([all_results_with_int.iImage]).classID]==posClass)-1;
res_scores = [all_results_with_int.bestConfigScore];
[sortedScores1,showOrder1] = sort(res_scores,'descend');
%delay=0,trues=[],displayTrue=0,indexto
% [sortedScores1,showOrder1] = sort(res_scores,'descend');
vl_pr(res_classes,res_scores)
% [sortedScores1,showOrder1] = sort(res_scores+face_scores,'descend');
trues = res_classes(showOrder1)==1;
displayImageSeries(conf,fra_db(f_test(showOrder1)),0,trues,0,sortedScores1)
vl_pr(test_labels,face_scores+.3*res_scores)
[sortedScores1,showOrder1] = sort(face_scores,'descend');
displayImageSeries(conf,fra_db(f_test(showOrder1)),0,trues,-1,sortedScores1)
vl_pr(test_labels,face_scores)
%% good, now create the image graph using some superpixel method...
% cur_set = f_train(randperm(length(f_train)));
cur_set = f_test_pos;
for it = 3:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    I = getImage(conf,imgData);
    min_gt_ovp = 0;
    %[configs,scores] = findBestConfiguration(imgData,I,regionSampler,params);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    % constrain search using the coarse method!
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    
    %z_factor = 4;
    % [candidates,ucm,segBox,isvalid] = getSegmentation(conf,imgData,true,'~/storage/fra_face_seg');
    %x2(ucm)
    %Z = cropper(I,segBox);
    %I = imResample(I,[maxImageSize maxImageSize]);
    conf.get_full_image = true;
    %for zFactor = [.5:.1:1.5]
    zFactor = .7;
    I = im2uint8(getImage(conf,imgData));
    F = inflatebbox(imgData.mouth,h*zFactor,'both',true);
    I = cropper(I,round(F));
    sal_opts.show = false;
    maxImageSize = 50;
    sal_opts.maxImageSize = maxImageSize;
    I = imResample(I,[maxImageSize maxImageSize]);
    Z = [];
    %sizeRatio = [10 15 20]
    sizeRatio = 10
    for curSizeRatio = sizeRatio
        spSize = (maxImageSize/curSizeRatio)^2;
        %     spSize = 50;
        %     I = imResample(I,size2(sal1),'bilinear');
        sal_opts.pixNumInSP = spSize;
        sal_opts.useSP = true;
        [sal1,sal_bd,resizeRatio,sp_data] = extractSaliencyMap(I,sal_opts);
        if isempty(Z)
            Z = sal1;
        else
            Z = Z+sal1/length(sizeRatio);
        end
    end
    %sal1 = double(sal1>0.1);
    I = imResample(I,size2(Z));
    sal1 = Z;
    I = im2double(I);
    clf;
    subplot(1,2,1);
    imagesc2(I);
    subplot(1,2,2);
    imagesc2(sc(cat(3,sal1,I),'prob_jet'));
    dpc
    %     end
    continue
    %     pause
    I = im2uint8(getImage(conf,imgData));
    [candidates, ucm2] = im2mcg(I,'accurate',true);
    I = imcrop(I);
    [candidates, ucm2] = im2mcg(I,'accurate',true);
    
    regions = row(squeeze(mat2cell2(candidates.masks,[1,1,size(candidates.masks,3)])));
    %
    region_scores = cellfun3(@(x) sum(sal1(x(:))),regions);
    region_areas = cellfun3(@(x) sum(x(:)),regions);
    region_scores = region_scores./region_areas;
    borderImage = addBorder(zeros(size2(I)),1,1);
    region_touch_border = cellfun3(@(x) sum(borderImage(x(:))),regions);
    region_scores(region_touch_border==0) = -1000;
    %sel_ = region_scores > .1;
    displayRegions(I,regions,region_scores);
    %
    %     I = imResample(Z,size2(regions{1}));
    regions = candidates.masks;
    [r,ro] = chooseRegion(I,regions,.5);
    displayRegions(I,r,ro);
    %
    %
    %
    %     break
end
%%
%it=
%t = cur_set(it);
t = 445
imgData = fra_db(t);
I = im2uint8(getImage(conf,imgData));
I = cropper(I,round(inflatebbox(imgData.faceBox,2,'both',false)));
close all
Z0 = zeros(size2(I));
for nPatches = [5 10 20]
    [ rects ] = makeTiles( I, nPatches);
    rects = clip_to_image(rects,I);
    % showSortedBoxes(I,rects)
    patches = multiCrop2(I,rects);
    close all
    %x2(patches);
    sals = {};
    for t = 1:length(patches)
        t/length(patches)
        maxImageSize = 50;
        sal_opts.maxImageSize = maxImageSize;
        II = imResample(patches{t},[maxImageSize maxImageSize]);
        Z = [];
        sizeRatio = 5;
        for curSizeRatio = sizeRatio
            spSize = (maxImageSize/curSizeRatio)^2;
            sal_opts.pixNumInSP = spSize;
            sal_opts.useSP = true;
            [sal1,sal_bd,resizeRatio,sp_data] = extractSaliencyMap(II,sal_opts);
            if isempty(Z)
                Z = sal1;
            else
                Z = Z+sal1/length(sizeRatio);
            end
        end
        sals{t} = Z;
    end
    x2(patches)
    x2(sals)
    
    Z = zeros(size2(I));
    for t = 1:length(patches)
        z = imResample(sals{t},size2(patches{t}));
        r = rects(t,:);
        Z(r(2):r(4),r(1):r(3)) = max(Z(r(2):r(4),r(1):r(3)),z);
    end
    
    Z0 = Z0+Z;
end
x2(sc(cat(3,Z,im2double(I)),'prob'))
x2(Z)


%%
% try to classify the shape dataset...
images = struct('imagePath',{},'image',{},'class',{},'descriptor',{});
D = getAllFiles('/home/amirro/data/mpeg7','gif');
% all_images = cellfun2(@imread,D);
all_images = {};
for t = 1:length(D)
    t
    I = imread(D{t});
    %     I = all_images{t};
    %     if length(size(I))==2
    %         I = repmat(I,[1 1 3]);
    %     end
    b = I > 0;
    b = cropper(b,region2Box(b));
    all_images{t} = b;
end

for t = 1:length(all_images)
    images(t).I =all_images{t};
    curPath = D{t};
    [~,name,~] = fileparts(curPath);
    images(t).class = name(1:strfind(name,'-')-1);
    images(t).imagePath = curPath;
    images(t).name = name;
end
image_classes = {images.class};
all_classes = unique(image_classes);
[lia,locb] = ismember(image_classes,all_classes);
find(lia==1)
image_classes(1:10)

displayImageSeries2(all_images(1:10:end))

imgs1 = all_images;
for t = 1:length(imgs1)
    I = all_images{t};
    if length(size(I))==2
        I = repmat(I,[1 1 3]);
    end
    imgs1{t}=im2uint8(I);
end
all_feats = featureExtractor.extractFeaturesMulti(imgs1,false);


sel_test = 1:4:length(image_classes);
sel_train = setdiff(1:length(image_classes),sel_test);
feats_train = all_feats(:,sel_train);
feats_test = all_feats(:,sel_test);
classes_train = locb(sel_train);
classes_test = locb(sel_test);
results = struct;

for iClass = 1:length(all_classes)
    clc
    iClass
    isClass_train = classes_train == iClass;
    p = Pegasos(feats_train,isClass_train'*2-1);
    w = p.w(1:end-1);
    b = p.w(end);
    isClass_test = classes_test==iClass;
    scores_test = w'*feats_test;
    [recall,precision,info] = vl_pr(isClass_test*2-1,scores_test);
    results(iClass).recall = recall;
    results(iClass).precision = precision;
    results(iClass).info = info;
    results(iClass).w = w;
    results(iClass).b = b;
    results(iClass).scores = scores_test;
    %showSorted(imgs1(sel_test),scores_test)
end

infos = [results.info];
aps = [infos.ap];
[r,ir] = sort(aps);
%%
for iu = 1:length(ir)
    % u = ir(iu)
    u = iu;
    curImgs = imgs1(sel_test);
    [z,iz] = sort(results(u).scores,'descend');
    iz = iz(1:50);
    curImgs = multiResize(curImgs(iz),[64 64]);
    clf;
    subplot(1,2,1);
    labels = 2*(classes_test==u)-1;
    curImgs = paintRule(curImgs,labels,[],[],4);
    imagesc2(mImage(curImgs));
    subplot(1,2,2); vl_pr(labels,results(u).scores);
    dpc;
end
%%
plot(r)
ir(1)
iClass = 4
%x(imgs1(sel_test(isClass_test)))
%displayImageSeries2(all_images)
%% classify small sub images to prove the point
% collect training data - these are all the image segments
fra_db = load_dlib_landmarks(conf,fra_db);
curSet = f_train;
all_data = struct('img',{},'regions',{},'gt_ovp',{});
n = 0;
goods = false(size(curSet));
dlib_landmark_split;
% training
%%
curSet = f_train_pos;




pos_patches = {};
neg_patches = {};
for it = 1:1:length(curSet)
    t = curSet(it);
    t
    imgData = fra_db(t);
    I =  getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    if (isempty(gt_graph))
        continue
    end
    if ~isfield(gt_graph{2},'roiMask')
        continue
    end
    [I_sub,mouthBox,candidates,isvalid] = getCandidateRegions(conf,imgData);
    if ~isvalid
        continue
    end
    
    x2(I);
    plot_dlib_landmarks(imgData.Landmarks_dlib);
    plotBoxes(imgData.faceBox);  
    [regions,mouthMask,roiMask] = processRegions(imgData,I_sub,candidates,mouthBox,gt_graph{2}.roiMask);
    
    [patches,labels] = sampleActionPatches(I_sub,mouthMask,roiMask,size(mouthMask,1)/3);
    
    perimBoxes = round(sampleOnPerimeter(mouthMask,size(mouthMask,1)/3));
    perimBoxes = clip_to_image(perimBoxes,mouthMask);
    curMask = roiMask &~mouthMask;
    [pos_samples,neg_samples] = sampleUsingRegion(perimBoxes,curMask);    
    Q = blendRegion(I_sub,double(mouthMask),.3,[0 1 0]);
%     clf; imagesc2(Q); dpc;continue    
%     clf; imagesc2(I_sub);
%     plotBoxes(pos_samples,'g-');
%     plotBoxes(neg_samples,'r-');
    
    pos_patches{end+1} = multiCrop2(I_sub,round(pos_samples));
    neg_patches{end+1} = multiCrop2(I_sub,round(neg_samples));
%     dpc; continue    
    continued
            
    Q = blendRegion(I_sub,double(mouthMask),.3,[0 1 0]);
    
    
%     x2(I_sub); plotBoxes(pos_samples);
%     plotBoxes(neg_samples,'r-');
    
    x2(Q)
    
    
    
    x2(roiMask)
    x2(Q)
    
    
    startPt = fliplr(size2(I_sub))/2;
    curParams = params.sampling;
    curParams.lengths = (size(I_sub,1));
    curParams.widths =  curParams.lengths/4;
    
    %     displayRegions(I_sub,roiMasks)
    
    
    curParams.thetas = params.sampling.thetas(1:end/2);
    rois1 = getCandidateRoisPolygons(startPt,1,curParams,true);
    roiMasks1 = cellfun2(@(x) poly2mask2(x.xy,size2(I_sub)), rois1);
    curParams.thetas = 360-params.sampling.thetas(1:end/2);
    rois2 = getCandidateRoisPolygons(startPt,1,curParams,true);
    roiMasks2 = cellfun2(@(x) poly2mask2(x.xy,size2(I_sub)), rois2);
    %
    %     for t = 1:length(roiMasks1)
    %         clf; imagesc2(roiMasks1{t}+roiMasks2{t});
    %         dpc
    %     end    
    % find assymetric pairs
    close all
    patches1 = {};
    patches2 = {};
    for u = 1:length(roiMasks1)
        patches1{u} = maskedPatch(I_sub,roiMasks1{u},true,.5);
        patches2{u} = maskedPatch(I_sub,roiMasks2{u},true,.5);
    end        
    %     patches2 = cellfun2(@(x) imrotate(flip_image(flip_image(x,1),2),180), patches2);    
    patches2 = cellfun2(@(x) flip_image(x,2), patches2);
    %     x2(patches2);x2(patches1);    
    feats1 = featureExtractor.extractFeaturesMulti(patches1,true);
    feats2 = featureExtractor.extractFeaturesMulti(patches2,true);    
    feats1 = normalize_vec(feats1);
    feats2 = normalize_vec(feats2);
    scores = sum((feats1.*feats2));
    %plot(sum(feats1.*feats2))
    %m = showSorted(patches2,-sum(feats1.*feats2),2);    
    clf; subplot(1,2,1);
    imagesc2(I_sub);
    subplot(1,2,2);
    S = zeros(size2(I_sub));
    for q=1:length(roiMasks1)
        S = max(S,double(roiMasks1{q} | roiMasks2{q})*(1-scores(q)));
    end    
    imagesc2(sc(cat(3,S,I_sub),'prob'));            
    dpc;continue
    r = [1 1 size2(I_sub)];
    inner_mask = round(inflatebbox(r,.6,'both',false));
    inner_mask = box2Region(inner_mask,size2(I_sub));
    %x2(I_sub); plotBoxes(inner_mask)
    %     regions = cellfun2(@(x) x.*inner_mask,regions);
    %     mouthMask = mouthMask.*inner_mask;
    %     roiMask = roiMask.*inner_mask;
    
    regions = removeDuplicateRegions(regions);
    displayRegions(Q,regions);
    
    dpc;continue;
    
    % regions should not occupy a large angular range.
    regions1 = cellfun2(@(x) imResample(single(x),[15 15]),regions);
    
    displayRegions(Q,regions);
    %clf; imagesc2(I_sub);
    if nnz(roiMask)==0
        continue
    end
    [ovps ints uns] = regionsOverlap3({roiMask}, regions);
    n = n+1;
    all_data(n).img = I_sub;
    all_data(n).regions = regions;
    all_data(n).gt_ovp = ovps;
    all_data(n).ind_orig = t;
    all_data(n).classID = imgData.classID;
    %all_data(n).class_id =
    goods(it) = true;
end
selectionParams.min_pos_gt = .5;
selectionParams.max_neg_gt = .1;
selectionParams.maxNegPerImage = 10;
% select subset of regions for training data....
all_data_train = getTrainingRegions(all_data,selectionParams);
%%
pos_class = 4; % brushing teeth
[trainingFeats,trainLabels] = extractFeats(all_data_train,featureExtractor);
trainingFeats = cat(1,trainingFeats.feats);


%[x,y] = featsToLabels(feats_pos_local,feats_neg_local);
Y = 2*(trainLabels == pos_class)-1;
p = Pegasos(trainingFeats,Y(:),'lambda',.01,'foldNum',5,'bias',1);
w = p.w(1:end-1);

%% testing
curSet = f_test;
% retain top 5 segments for each image.
res = struct;
for it = 1:length(curSet)
    res(it).isvalid = false;
    t = curSet(it);
    t
    imgData = fra_db(t);
    I =  getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    [I_sub,mouthBox,candidates,isvalid] = getCandidateRegions(conf,imgData);
    if ~isvalid
        continue
    end
    [regions,mouthMask,roiMask] = processRegions(imgData,I_sub,candidates,mouthBox,gt_graph{2}.roiMask);
    clear curData; curData.img = I_sub; curData.regions = regions;
    curData.classID = imgData.classID;
    curData.labels = [];
    curFeats = extractFeats(curData,featureExtractor);
    F = cat(1,curFeats.feats);
    curScores = w'*F;
    res(it).isvalid = true;
    res(it).imgData = imgData;
    [r,ir] = sort(curScores,'descend');
    ir = ir(1:min(5,length(ir)));
    r = r(1:min(5,length(r)));
    regions = regions(ir);
    res(it).regions = regions;
    res(it).scores = r;
    %     displayRegions(I_sub,regions,r,0,5);
end

%
%%
scores = zeros(size(res));
for t = 1:length(res)
    if res(t).isvalid==0
        scores(t) = -200;
    else
        scores(t) = max(res(t).scores);
    end
end

set_labels = isClass(curSet)*2-1;
vl_pr(set_labels,scores)
[v,iv] = sort(scores,'descend');
%displayImageSeries(conf,fra_db(curSet(iv)),0)
%%
for it = 1:length(scores)
    100*it/length(scores)
    t = iv(it)
    curData = res(t);
    if curData.isvalid==false
        continue
    end
    imgData = curData.imgData;
    if imgData.classID==4,continue,end
    I = getImage(conf,imgData);
    faceBox = imgData.faceBox;
    mouthCenter = imgData.mouth;
    h = faceBox(3)-faceBox(1);
    mouthBox = round(inflatebbox(mouthCenter,h,'both',true));
    I_sub = cropper(I,mouthBox);
    %  [I_sub,mouthBox,candidates,isvalid] = getCandidateRegions(conf,imgData);
    displayRegions(I_sub,curData.regions,curData.scores)
    
    
end
%%
%% quantify the amount of overlap for each training example.
ovps = zeros(size(all_data));
for t = 1:length(all_data)
    curOvp = all_data(t).gt_ovp(1:end-1);
    ovps(t) = max(curOvp);
    continue
    curRegions = all_data(t).regions(1:end-1);
end
[u,iu] = sort(ovps,'ascend');

[no,xo] = hist(ovps,10);
figure,
bar(xo,no/sum(no));xlabel('best overlap'); ylabel('n.images');
title('MCG overlap with ground truth')

%%
for it = 15:length(ovps)
    t = iu(it);
    curRegions = all_data(t).regions(1:end-1);
    curOvps = all_data(t).gt_ovp(1:end-1);
    [z,iz] = sort(curOvps,'descend');
    clf;
    I = im2double(all_data(t).img);
    vl_tightsubplot(1,3,1);
    imagesc2(I);title('img')
    vl_tightsubplot(1,3,2);
    Q = displayRegions(I,all_data(t).regions(end));
    Q = addBorder(Q,1,[0 1 0]);
    imagesc2(Q);title('ground truth');
    vl_tightsubplot(1,3,3);
    displayRegions(I,curRegions,curOvps,0,10);
    %     title(num2str(z(1)));
    continue
    % % %
    % %     [regions,ovp,sel_] = chooseRegion(I,curRegions,.5);
    % %
    % %
    % %     [cands,ucm2] = im2mcg(I,'accurate',true);
    % %     masks= row(squeeze(mat2cell2(cands.masks,[1,1,size(cands.masks,3)])));
    % %
    % %     segs = vl_slic(im2single(I), 20, .01);
    % %
    %     masks2 = {}
    %     for t = 1:max(segs(:))
    %         masks2{t} = segs==t;
    %     end
    % %
    % %     x2(paintSeg(I,segs))
    % %     [regions,ovp,sel_] = chooseRegion(I,masks,.5);
    % %     displayRegions(I,regions,ovp);
    
    dpc
end
