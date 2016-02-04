
% 6/5/2015cur_set = f_test_pos;
theta_start = 0;
nSamples = 25;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
% cur_set = cur_set(randperm(length(cur_set)));
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
%     gt_graph = get_gt_graph(imgData,nodes,params,I);
%     g = poly2mask2(cellfun2(@(x) x.poly,gt_graph),size2(I));
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    avgWidth = .5*h;
    avgLength = 1.5*h;
    startPt = boxCenters(faceBox);
    rois = hingedSample(startPt,avgWidth,avgLength,thetas);
    roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,x.xy,[avgLength avgWidth]),rois);
    curFeats = featureExtractor.extractFeaturesMulti(roiPatches);
    curScores = w_int'*curFeats+b_int;
    [r,ir] = sort(curScores,'descend');
    S = computeHeatMap_regions(I,roiMasks,normalise(curScores).^2,'max');
%     S = computeHeatMap_regions(I,roiMasks(ir(1)),normalise(curScores(ir(1))),'max');
    figure(1); clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2);
    imagesc2(sc(cat(3,S,im2double(I)),'prob'));
    title(num2str(max(curScores)));
    dpc
%     [~,ints,uns] = regionsOverlap(roiMasks,g);
%     ints = ints./cellfun(@nnz,roiMasks)';    
%     roi_pos_patches{end+1} = roiPatches(ints > .5);
%     roi_neg_patches{end+1} = roiPatches(ints < .1); 
end

%% stage 1: object graphs
cur_set = f_train_pos;
toVisualize = false;

%---------
% Action Object Detection - a structured output framework for action object detection
% and description.
initpath;
config;
addpath('/home/amirro/code/3rdparty/edgeBoxes/');
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
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

edges(1).v = [1 2];
edges(1).connection = 'anchor';
edges(2).v = [2 3];
edges(2).connection = 'anchor';
for iNode = 1:length(nodes)
    nodes(iNode).valid = true;
end
% sample good configurations, configurations are assignments to the graph,
% and each configuration is defined a set of polygons defining image
% regions
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
params.cand_mode = 'edgeBoxes';
imgData = fra_db(1);
x2(getImage(conf,imgData))
plotBoxes(imgData.faceBox)
regionSampler = RegionSampler();
regionSampler.clearRoi();
%%
featureExtractor = DeepFeatureExtractor(conf);
% featureExtractorLite = DeepFeatureExtractor_nin(conf);
% F = featureExtractorLite.extractFeatures(I,bb(:,1:4));
%% stage -1 training: train region-of-interaction detector
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
for it = 1:length(cur_set)
    it
%     profile on
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);    
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    g = poly2mask2(cellfun2(@(x) x.poly,gt_graph),size2(I));   
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    avgWidth = .5*h;
    avgLength = 1.5*h;
    startPt = boxCenters(faceBox);    
    rois = hingedSample(startPt,avgWidth,avgLength,thetas);
    roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,round(x.xy),[avgLength avgWidth]),rois);
    [~,ints,uns] = regionsOverlap(roiMasks,g);
    ints = ints./cellfun(@nnz,roiMasks)';    
    roi_pos_patches{end+1} = roiPatches(ints > .5);
    roi_neg_patches{end+1} = roiPatches(ints < .1); 
%     profile viewer
end

% z = getLogPolarMask(50,10,2);

%%
roi_pos_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_pos_patches{:}));
roi_neg_feats = featureExtractor.extractFeaturesMulti(cat(2,roi_neg_patches{:}));
%%
[w_int b_int] = concat_and_learn(roi_pos_feats,roi_neg_feats);
%%
%% show some stage -1 results.
cur_set = f_test_pos;
theta_start = 0;
nSamples = 25;
theta_end = theta_start+360;
b = (theta_end-theta_start)/nSamples;
theta_end = theta_end-b;
thetas = theta_start:b:theta_end;%0:10:350;
% cur_set = cur_set(randperm(length(cur_set)));
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
%     gt_graph = get_gt_graph(imgData,nodes,params,I);
%     g = poly2mask2(cellfun2(@(x) x.poly,gt_graph),size2(I));
    faceBox = imgData.faceBox;
    h = faceBox(4)-faceBox(2);
    avgWidth = .5*h;
    avgLength = 1.5*h;
    startPt = boxCenters(faceBox);
    rois = hingedSample(startPt,avgWidth,avgLength,thetas);
    roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,x.xy,[avgLength avgWidth]),rois);
    curFeats = featureExtractor.extractFeaturesMulti(roiPatches);
    curScores = w_int'*curFeats+b_int;
    [r,ir] = sort(curScores,'descend');
    S = computeHeatMap_regions(I,roiMasks,normalise(curScores).^2,'max');
%     S = computeHeatMap_regions(I,roiMasks(ir(1)),normalise(curScores(ir(1))),'max');
    figure(1); clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2);
    imagesc2(sc(cat(3,S,im2double(I)),'prob'));
    title(num2str(max(curScores)));
    dpc
%     [~,ints,uns] = regionsOverlap(roiMasks,g);
%     ints = ints./cellfun(@nnz,roiMasks)';    
%     roi_pos_patches{end+1} = roiPatches(ints > .5);
%     roi_neg_patches{end+1} = roiPatches(ints < .1); 
end

%% stage 1: object graphs
cur_set = f_train_pos;
toVisualize = false;
all_pos_feats = {};
all_pos_feats_light = {};
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);    
    gt_graph = get_gt_graph(imgData,nodes,params,I);
%     clf; x2(I); 
    min_gt_ovp = .5; % .35
    
    
    gt_configurations = sample_configurations(imgData,I,min_gt_ovp,gt_graph,regionSampler,params);    
    all_pos_feats{end+1} = configurationToFeats2(I,gt_configurations,featureExtractor);
    %all_pos_feats{end+1} = configurationToFeats(I,boxes,routes,featureExtractor);
    
    
    %[boxes,routes] = sampleGraph_w_edgeBoxes(imgData,I,min_gt_ovp,gt_graph,regionSampler);
    
    if size(routes,1) > 5
        routes = routes(1:5,:);
    end
    
    configurations =  {};
    all_pos_feats{end+1} = configurationToFeats(I,boxes,routes,featureExtractor);
    
    
    %     
%          for t = 1:size(routes,1)
%             clf;
%             imagesc2(I);
%             plotBoxes(boxes(routes(t,:),:));
%             dpc
%         end
    
% %     continue
% %     if toVisualize
% %         clf; imagesc2(I);
% %         p = gt_graph;
% %         polys = cellfun2(@(x) x.poly,p);
% %         plotPolygons(polys,'g-');
% %         dpc
% %     end
    
% 
% configurations =  {};
% for t = 1:size(routes,1)
%     configurations{t} = boxes(routes(t,:),:);
% end

%     all_pos_feats_light{end+1} = configurationToFeats(I,posConfig,nodes,featureExtractorLite);
end
%% negative samples
regionSampler.clearRoi();
cur_set = vl_colsubset(row(f_train_neg),100,'Uniform');
all_neg_feats = {};
all_neg_feats_light = {};
toVisualize=false;
for it = 1:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);
    opts.donms = false;
    boxes = regionSampler.sampleEdgeBoxes(I);
    opts.donms = 0;
    [boxes,boxGraph] = getBoxGraphForImage(imgData,I,gt_graph,boxes,opts);
    % sample configurations from graph
    routeLength = length(gt_graph);
    nRoutes = 5;
    routes = sampleRoutes(boxGraph,routeLength,nRoutes);
    configurations =  {};
%     for t = 1:size(routes,1)
%         configurations{t} = boxes(routes(t,:),:);
%     end
    % % %     for t = 1:size(routes,1)
    % % %         clf;
    % % %         imagesc2(I);
    % % %         plotBoxes(boxes(routes(t,:),:));
    % % %         dpc
    % % %     end
    %negConfig = sampleGraph(imgData,false,nodes,edges,params,I,regionSampler,5);
    %     negConfig = sampleGraph(imgData,false,nodes,edges,params,I,regionSampler,5);
    %     [candidate_boxes,routes] = sampleGraph_w_edgeBoxes(imgData,I,0,gt_graph,regionSampler);
    %     continue;
    
    all_neg_feats{end+1} = configurationToFeats(I,boxes,routes,featureExtractor);
    %     if toVisualize
    %         for u = 1:length(negConfig)
    %             clf; imagesc2(I);
    %             p = negConfig{u};
    %             polys = cellfun2(@(x) x.poly,p);
    %             plotPolygons(polys,'g-');
    %             dpc
    %         end
    %     end
end
%%
% train a classifier
%%
% zz_pos = cat(2,all_pos_feats{:});
% lengths = cellfun(@length,zz_pos);
% zz_pos = cat(2,zz_pos{lengths==max(lengths)});
% zz_neg = cat(2,all_neg_feats{:});
% lengths = cellfun(@length,zz_neg);
% zz_neg = cat(2,zz_neg{:});
% [x,y] = featsToLabels(zz_pos,zz_neg);
% [w b info] = vl_svmtrain(x, y, .00001);

% save ~/storage/data/tmp.mat all_pos_feats all_neg_feats
[w b info] = concat_and_learn(all_pos_feats,all_neg_feats);
% % [w_light b_light info_light] = concat_and_learn(all_pos_feats_light,all_neg_feats_light);

%%
% cur_set = f_test_pos;
toVisualize = false;
close all
regionSampler.clearRoi();
mmm = f_test_neg;
cur_set = vl_colsubset(row(mmm),100,'Uniform');
all_neg_feats = {};
all_neg_feats_light = {};
toVisualize=false;
figure(1);
for it = 7:length(cur_set)
    it
    t = cur_set(it);
    imgData = fra_db(t);
    imgData.imageID
    I = getImage(conf,imgData);
    gt_graph = get_gt_graph(imgData,nodes,params,I);    
    opts.donms = true;
    opts.nmsfactor = .8;
        
    startPt = boxCenters(imgData.faceBox);
    % first, apply the region-of-interest detection
    rois = hingedSample(startPt,avgWidth,avgLength,thetas);
    roiMasks = cellfun2(@(x) poly2mask2(x.xy,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,x.xy,[avgLength avgWidth]),rois);
    curFeats = featureExtractor.extractFeaturesMulti(roiPatches);
    curScores = w_int'*curFeats+b_int;
    [r,ir] = sort(curScores,'descend');
    S = computeHeatMap_regions(I,roiMasks(ir(1:3)),normalise(curScores(ir(1:3))),'max');    
    predicted_roi = max(cat(3,roiMasks{ir(1:3)}),[],3);        
    clf; 
    mm = 2;
    nn = 2;
    subplot(2,2,1); imagesc2(I); title('orig');
    subplot(2,2,2); imagesc2(sc(cat(3,S,I),'prob')); title('interaction');
    
    regionSampler.clearRoi();
    regionSampler.borders = [];
    boxes = regionSampler.sampleEdgeBoxes(I);
    boxes = boxes(:,1:4);
    % remove all boxes not touching the roi
    sums = sum_boxes(double(predicted_roi),boxes);
    areas = sum_boxes(ones(size2(I)),boxes);
    probs = sums./areas;
    boxes(probs<.1,:) = [];
    probs(probs<.1) = [];
%         
    opts.donms = 0;
    [boxes,boxGraph] = getBoxGraphForImage(imgData,I,gt_graph,boxes,opts);
    
    % sample configurations from graph
    routeLength = length(gt_graph);
    nRoutes = 30;
    withMemory = true;
    routes = sampleRoutes(boxGraph,routeLength,nRoutes);    
    bads = findWeirdRoutes(routes,boxes);
    routes = routes(~bads,:);

    configurations =  {};    
    curFeats = configurationToFeats(I,boxes,routes,featureExtractor);
    curFeats = cat(2,curFeats{:});
    curScores = w'*curFeats+b;
    [r,ir] = sort(curScores,'descend');
    for k = 1:5%size(routes,1)
        t = ir(k);
        subplot(mm,nn,3); gca;
        imagesc2(I);
        plotBoxes(boxes(routes(t,:),:));
        title(num2str(r(k)));
        dpc
    end       
end


%% stage 0: train mouth with object / mouth with no object
%% stage 1: train a point-of-contact detector
%%
% rect type 0: ok
% rect type -1: out of border or "don't care" (small overlap with roi)
% rect type 1: overlap roi
% get positive / negative samples
regionSampler = RegionSampler();
regionSampler.boxOverlap = 0.3;
regionSampler.boxSize = [1 1]*32;
regionSampler.minRoiOverlap = .5;
mouth_rect_ratio = .5*[.3 .2];
%
pos_interaction_samples = {};
neg_interaction_samples = {};
doDebugging = true;
for it = 1:length(f_train_pos)
    it
    t = f_train_pos(it);
    if (~all(fra_db(t).face_landmarks.valids))
        it
        continue
    end
    t_orig = findImageIndex(fra_db,fra_db(t).imageID);
    [samples,labels] = sampleAroundMouth(fra_db(t),regionSampler,true);
    I = fra_db(t).I;
    if doDebugging
        clf; imagesc2(I);
        plotBoxes(samples(labels==0,:),'g-')
        plotBoxes(samples(labels==1,:),'r-');
        dpc
    end
    pos_sample_boxes = samples(labels==1,:);
    if (~isempty(pos_sample_boxes))
        %         pos_sample_boxes = round(inflatebbox(pos_sample_boxes,2,'both',false));
        pos_sample_centers = boxCenters(pos_sample_boxes);
        mouthCenter = fra_db(t).face_landmarks.xy(3,:);
        d_to_center = l2(mouthCenter,pos_sample_centers);
        [~,id] = min(d_to_center);
        pos_interaction_samples{end+1} = {cropper(I,pos_sample_boxes(id,:))};
        %     if (~isempty(p))
        %         pos_interaction_samples{end+1} = p(1);
        %     end
    end
    %     else
    %         break
    %     end
    
    neg_interaction_samples{end+1} = multiCrop2(I,samples(labels==0,:));
    %
end
%
% montage3(pos_interaction_samples)
pos_interaction_samples = cat(2,pos_interaction_samples{:});
%%
pos_interaction_samples = [pos_interaction_samples,flipAll(pos_interaction_samples)];
neg_interaction_samples = cat(2, neg_interaction_samples{:});
% showCoords(mouthCorners);
%%
% extract various feature types for each element:
featureExtractor_cifar = DeepFeatureExtractor_cifar(conf);
% featureExtractor = DeepFeatureExtractor(conf);
pos_feats_train = featureExtractor_cifar.extractFeaturesMulti(pos_interaction_samples);
neg_feats_train = featureExtractor_cifar.extractFeaturesMulti(neg_interaction_samples);

[x,y] = featsToLabels(pos_feats_train,neg_feats_train);
% x = vl_homkermap(x,1);
[w_int b_int info] = vl_svmtrain(x, y, .001);

%% show on some positive/negative test examples:
% set_ =  \;
set_ = set_(randperm(length(set_)));
regionSampler.boxOverlap = 0.5;

%%
for it = 1:length(set_)
    t = set_(it);
    if (~all(fra_db(t).face_landmarks.valids))
        continue
    end
    I = fra_db(t).I;
    [s,labels] = sampleAroundMouth(fra_db(t),regionSampler,false);
    %s(:,1:4) = round(s(:,1:4));
    boxLabels = s(:,end);
    patches = multiCrop2(I,s);
    curFeatsTest = featureExtractor_cifar.extractFeaturesMulti(patches, false);
    curScores = w_int'*curFeatsTest+b_int;
    [z,counts] = computeHeatMap(I,[s curScores(:)],'max');
    clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2);
    imagesc2(sc(cat(3,z,im2double(I)),'prob'));
    title(num2str(max(curScores)));
    dpc;
end

%% stage 2: train an object detector; first apply some learned geometric rules,
%% then use appearance
pos_boxes = {};
neg_boxes = {};
for iu = 1:length(f_train_pos)
    iu
    t = f_train_pos(iu);
    imgData = fra_db(t);
    %     x2(imgData.I); showCoords(imgData.face_landmarks.xy)
    mouthCenter = imgData.face_landmarks.xy(3,:);
    assert(all(imgData.face_landmarks.valids(3)))
    %     continue
    I = imgData.I;
    sz = size2(I);
    obj_poly = imgData.gt_obj;
    obj_box = pts2Box(cat(1,obj_poly{:}));
    obj_box = BoxIntersection(obj_box,[1 1 fliplr(sz)]);
    regionSampler.clearRoi();
    [edgeBoxSamples] = regionSampler.sampleEdgeBoxes(I);
    [~,~,a] = BoxSize(edgeBoxSamples);
    [ovps,ints] = boxesOverlap(edgeBoxSamples,obj_box);
    f_neg = vl_colsubset(row(find(ints==0)),10,'Uniform');
    pos_boxes{end+1} = box_point_interaction(obj_box,I,mouthCenter);
    neg_boxes{end+1} = box_point_interaction(edgeBoxSamples(f_neg,1:4),I,mouthCenter);
    %     x2(I); plotBoxes(edgeBoxSamples(f_neg,:));
end

pos_boxes = cat(2,pos_boxes{:});
neg_boxes = cat(2,neg_boxes{:});

% box_pos_feats_train =  pos_boxes/192;
% box_neg_feats_train = neg_boxes/192;

[x,y] = featsToLabels(pos_boxes,neg_boxes);
% x = vl_homkermap(x,1);
% x = vl_homkermap(x,1);
%%
[w_box, b_box, info] = vl_svmtrain(x, y, .0001);
%% try it
close all
for iu = 1:length(f_test_pos);
    iu
    t = f_test_pos(iu);
    imgData = fra_db(t);
    I = imgData.I;
    sz = size2(I);
    regionSampler.clearRoi();
    edgeBoxSamples = regionSampler.sampleEdgeBoxes(I);
    
    edgeBoxSamples = edgeBoxSamples(:,1:4);
    %     curFeats = edgeBoxSamples'/192;
    %curFeats = vl_homkermap(curFeats,1);
    mouthCenter = imgData.face_landmarks.xy(3,:);
    curFeats = box_point_interaction(edgeBoxSamples(:,1:4),I,mouthCenter);
    curScores = w_box'*curFeats+b_box;
    %     curScores = ones(size(curScores));
    toDiscard = curScores < 0;
    if (all(toDiscard))
        [r,ir] = sort(curScores,'descend');
        toDiscard(ir(1:min(100,length(ir)))) = false;
    end
    edgeBoxSamples(toDiscard,:) = [];
    
    
    curScores(toDiscard) = [];
    [r,ir] = sort(curScores,'descend');
    %     for iBox = 1:4
    %         clf; imagesc2(I);
    %         plotBoxes(edgeBoxSamples(ir(iBox),:))
    %         dpc(.1)
    %     end
    obj_poly = imgData.gt_obj;
    obj_box = pts2Box(cat(1,obj_poly{:}));
    obj_box = BoxIntersection(obj_box,[1 1 fliplr(sz)]);
    [ovps,ints] = boxesOverlap(edgeBoxSamples,obj_box);
    %     clf; plot(curScores,ovps,'r+');
    %     dpc;
    %     continue
    fprintf('remaining samples:%d, best overlap: %f\n',length(ovps),max(ovps));
    
    curScores = normalise(curScores);
    z = computeHeatMap(I,[edgeBoxSamples curScores(:)],'max');
    clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2); imagesc2(sc(cat(3,z,im2double(I)),'prob'));
    
    dpc;
end
% ok, that looks pretty good, and on the other hand doesn't look like
% overfitting. We can discard obvious candidates from the *training set*
% using this method and compute appearance scores on them only.

%%
pos_windows = {};
neg_windows = {};
for iu = 1:length(f_train_pos)
    iu
    t = f_train_pos(iu);
    imgData = fra_db(t);
    I = imgData.I;
    sz = size2(I);
    obj_poly = imgData.gt_obj;
    obj_box = pts2Box(cat(1,obj_poly{:}));
    obj_box = BoxIntersection(obj_box,[1 1 fliplr(sz)]);
    regionSampler.clearRoi();
    [edgeBoxSamples] = regionSampler.sampleEdgeBoxes(I);
    [ovps,ints] = boxesOverlap(edgeBoxSamples,obj_box);
    edgeBoxSamples = edgeBoxSamples(ovps==0,1:4);
    mouthCenter = imgData.face_landmarks.xy(3,:);
    curFeats = box_point_interaction(edgeBoxSamples(:,1:4),I,mouthCenter);
    curScores = w_box'*curFeats+b_box;
    %     curScores = ones(size(curScores));
    toDiscard = curScores < 0;
    if (all(toDiscard))
        [r,ir] = sort(curScores,'descend');
        toDiscard(ir(1:min(5,length(ir)))) = false;
    end
    edgeBoxSamples(toDiscard,:) = [];
    
    f_neg = vl_colsubset(row(find(ints==0)),10,'Uniform');
    pos_windows{iu} = (cropper(I,round(obj_box)));
    neg_windows{iu} = multiCrop2(I,edgeBoxSamples);
    %     x2(I); plotBoxes(edgeBoxSamples(f_neg,:));
end

neg_windows = cat(2,neg_windows{:});
% neg_windows = cellfun2(@,neg_windows);
% featureExtractor = DeepFeatureExtractor_cifar(conf);
featureExtractor = DeepFeatureExtractor(conf);

win_pos_feats_train = featureExtractor.extractFeaturesMulti(pos_windows);
win_neg_feats_train = featureExtractor.extractFeaturesMulti(neg_windows);

[x,y] = featsToLabels(win_pos_feats_train,win_neg_feats_train);
% x = vl_homkermap(x,1);
[w_win b_win info] = vl_svmtrain(x, y, .0001);
%% show some performance on the object proposals, after removing some boxes
% using the geometric considerations
%% full model:
%1. filter object proposals using geometric constraints
%2. score object proposals using own appearance
%3. show for each proposal best interaction feature within its zone

set_ = f_test_pos;
for iu = 2:length(set_)%                             boxRegions{t} = box2Region(samples(t,:),obj.roi);
    iu
    t = set_(iu);
    imgData = fra_db(t);
    I = imgData.I;
    sz = size2(I);
    regionSampler.clearRoi();%                             boxRegions{t} = box2Region(samples(t,:),obj.roi);
    % 1. box candidates from image
    objectCandidates = regionSampler.sampleEdgeBoxes(I);
    objectCandidates = objectCandidates(:,1:4);
    mouthCenter = imgData.face_landmarks.xy(3,:);
    geomFeats = box_point_interaction(objectCandidates(:,1:4),I,mouthCenter);
    geomScores = w_box'*geomFeats+b_box;
    toDiscard = geomScores < 0;
    if (all(toDiscard))
        [r,ir] = sort(geomScores,'descend');
        toDiscard(ir(1:min(10,length(ir)))) = false;
    end
    objectCandidates(toDiscard,:) = [];
    geomScores(toDiscard) = [];
    objPatches = multiCrop2(I,objectCandidates);
    %     objPatches = cellfun2(@makeImageSquare,objPatches);
    objFeats = featureExtractor.extractFeaturesMulti(objPatches);
    objScores = w_win'*objFeats+b_win;
    
    regionSampler.boxOverlap = .2;
    [int_boxes] = sampleAroundMouth(fra_db(t),regionSampler,false);
    
    intPatches = multiCrop2(I,int_boxes);
    intAppFeats = featureExtractor_cifar.extractFeaturesMulti(intPatches, false);
    intScores = w_int'*intAppFeats+b_int;
    %     intPatches =
    
    intCenters = boxCenters(int_boxes);
    
    % computer for each region its best interactor
    touchingMatrix = zeros(length(objPatches),length(intPatches));
    for iR = 1:length(objPatches)
        touchingMatrix(iR,:) = inBox(objectCandidates(iR,:),intCenters);
        %         figure(1); clf; hold on; plotBoxes(objectCandidates(iR,:));
        %         plotPolygons(intCenters(touchingMatrix(iR,:),:),'m+');
        %         plotPolygons(intCenters(~touchingMatrix(iR,:),:),'r.');
        %         dpc
    end
    
    [ii,oo] = meshgrid(intScores,objScores);
    
    jointScore = ii+oo;
    jointScore(~touchingMatrix) = -inf;
    
    [join_obj,im] = max(jointScore,[],2);
    
    [~,iPoint] = max(join_obj);
    %z = computeHeatMap(I,[objectCandidates normalise(objScores(:))],'max');
    
    %     showSortedBoxes(I,objesctCandidates,join_obj)
    
    z = computeHeatMap(I,[objectCandidates normalise(join_obj(:))],'max');
    clf;
    subplot(2,2,1); imagesc2(I);title('orig');
    subplot(2,2,2); imagesc2(sc(cat(3,z,im2double(I)),'prob'));
    plotBoxes(int_boxes(ip,:));
    title(num2str(max(objScores)));
    z1 = computeHeatMap(I,[int_boxes normalise(intScores(:))],'max');
    subplot(2,2,3); imagesc2(sc(cat(3,z1,im2double(I)),'prob'));
    dpc;
    continue
    
    
        sal_opts.show = false;
        maxImageSize = 200;
        sal_opts.maxImageSize = maxImageSize;
        spSize = 25;
        sal_opts.pixNumInSP = spSize;
        conf.get_full_image = true;
        I = im2uint8(getImage(conf,imgData));
        [sal1,sal_bd,resizeRatio] = extractSaliencyMap(I,sal_opts);
    %
    %     sal = imResample(cropper(imResample(sal1,size2(I_orig)),imgData.roiBox),size2(I));
    %
    %     x2(I);x2(sal);
    
    sums = sum_boxes(sal,objectCandidates);
    areas = sum_boxes(ones(size2(sal)),objectCandidates);
    probs = sums./areas;
    
    close all;
    showSortedBoxes(I,objectCandidates,probs)
    showSortedBoxes(I,objectCandidates,objScores)
    
    z = computeHeatMap(I,[objectCandidates normalise(objScores(:))],'max');
    clf;
    subplot(1,2,1); imagesc2(I);
    subplot(1,2,2); imagesc2(sc(cat(3,z,im2double(I)),'prob'));
    title(num2str(max(objScores)));
    %     dpc;
end


%% alternative 4 : given a mouth center, train positive, negative sample for entire object, using oriented wedges
% first, do a statistic of the avg. object length

avgLength = 60;
avgWidth = 20;
pos_patches = {};
neg_patches = {};
for iu = 1:length(f_train_pos)
    iu
    t = f_train_pos(iu);
    imgData = fra_db(t);
    I = imgData.I;
    
    startPt = imgData.face_landmarks.xy(3,:);
    [rois,thetas] = hingedSample(startPt,avgWidth,avgLength,0:15:360);
    
    obj_poly = imgData.gt_obj;
    gt_region = poly2mask2(obj_poly,size2(I));
    roiMasks = cellfun2(@(x) poly2mask2(x,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,x,[avgLength avgWidth]),rois);
    [~,ints,uns] = regionsOverlap(roiMasks,gt_region);
    
    areas = cellfun(@nnz,roiMasks);
    ints = ints./areas(:);
    sel_pos = ints > .3;
    sel_neg = ints < .1;
    sel_neg = vl_colsubset(row(find(sel_neg)),5);
    pos_patches{end+1} = roiPatches(sel_pos);
    neg_patches{end+1} = roiPatches(sel_neg);
    
    % % %     clf; subplot(1,2,1); imagesc2(I); plotPolygons(rois(sel_pos),'g-+');
    % % %     subplot(1,2,2); imagesc2(I); plotPolygons(rois(sel_neg),'m-+');
    % % %     dpc
    %     displayRegions(I,roiMasks,ints,0);
    %     [ovp,ints,uns] = regionsOverlap(rr,gt_region);
    %
    %     displayRegions(I,rr,ovp,0,3);
    %
    %     continue
    % %     roi = directionalROI_(I,startPt,vec,roiWidth)
    %
    %     clf; displayRegions(I,gt_region);
    %     plotPolygons(imgData.face_landmarks.xy,'g.');
end
pos_patches = cat(2,pos_patches{:});
neg_patches= cat(2,neg_patches{:});
%%
pos_patches1 = [pos_patches,flipAll(pos_patches,1)];
% featureExtractor_cifar = DeepFeatureExtractor_cifar(conf);
featureExtractor = DeepFeatureExtractor(conf);

pos_feats_train = featureExtractor.extractFeaturesMulti(pos_patches1);
neg_feats_train = featureExtractor.extractFeaturesMulti(neg_patches);
%%
[x,y] = featsToLabels(pos_feats_train,neg_feats_train);
% x = vl_homkermap(x,1);
[w_wedge b_wedge info] = vl_svmtrain(x, y, .1);


%% try it!
% avgLength = 60;
% avgWidth = 20;
set_ = f_test_pos;
set_ = set_(randperm(length(set_)));
for iu = 1:length(set_)
    
    iu
    t = set_(iu);
    imgData = fra_db(t);
    I = imgData.I;
    startPt = imgData.face_landmarks.xy(3,:);
    
    %     [samples,labels] = sampleAroundMouth(imgData,regionSampler,false);
    %     samples = boxCenters(samples);
    %     all_rois = {};
    %     for tt = 1:size(samples,1)
    %         all_rois{tt} = hingedSample(samples(tt,:),avgWidth,avgLength,0:30:360);
    %     end
    %     rois = cat(2,all_rois{:});
    [rois,thetas] = hingedSample(startPt,avgWidth,avgLength,0:15:360);
    
    obj_poly = imgData.gt_obj;
    gt_region = poly2mask2(obj_poly,size2(I));
    roiMasks = cellfun2(@(x) poly2mask2(x,size2(I)),rois);
    roiPatches = cellfun2(@(x) rectifyWindow(I,x,[avgLength avgWidth]),rois);
    [~,ints,uns] = regionsOverlap(roiMasks,gt_region);
    curFeats=featureExtractor.extractFeaturesMulti(roiPatches);
    curScores = w_wedge'*curFeats+b_wedge;
    curScores1 = normalise(curScores).^2;
    S = computeHeatMap_regions(I,roiMasks,curScores1,'max');
    figure(5)
    clf;
    subplot(2,1,1); imagesc2(I);
    subplot(2,1,2);
    imagesc2(sc(cat(3,S,im2double(I)),'prob'));
    title(num2str(max(curScores)));
    dpc
end

%%
pos_windows = {};
neg_windows = {};
for iu = 1:length(f_train_pos)
    iu
    t = f_train_pos(iu);
    imgData = fra_db(t);
    I = imgData.I;
    sz = size2(I);
    obj_poly = imgData.gt_obj;
    obj_box = pts2Box(cat(1,obj_poly{:}));
    obj_box = BoxIntersection(obj_box,[1 1 fliplr(sz)]);
    regionSampler.clearRoi();
    [edgeBoxSamples] = regionSampler.sampleEdgeBoxes(I);
    [ovps,ints] = boxesOverlap(edgeBoxSamples,obj_box);
    edgeBoxSamples = edgeBoxSamples(ovps==0,1:4);
    mouthCenter = imgData.face_landmarks.xy(3,:);
    curFeats = box_point_interaction(edgeBoxSamples(:,1:4),I,mouthCenter);
    curScores = w_box'*curFeats+b_box;
    %     curScores = ones(size(curScores));
    toDiscard = curScores < 0;
    if (all(toDiscard))
        [r,ir] = sort(curScores,'descend');
        toDiscard(ir(1:min(5,length(ir)))) = false;
    end
    edgeBoxSamples(toDiscard,:) = [];
    f_neg = vl_colsubset(row(find(ints==0)),10,'Uniform');
    pos_windows{iu} = (cropper(I,round(obj_box)));
    neg_windows{iu} = multiCrop2(I,edgeBoxSamples);
    
    %     x2(I); plotBoxes(edgeBoxSamples(f_neg,:));
end




%% stage 3: find object proposals incident on prominent points of contact
for it = 1:length(f_train_pos)
    it
    t = f_train_pos(it);
    if (~all(fra_db(t).face_landmarks.valids))
        continue
    end
    I = fra_db(t).I;
    imgData = fra_db(t);
    obj_poly = imgData.gt_obj;
    obj_box = pts2Box(cat(1,obj_poly{:}));
    
    regionSampler.clearRoi();
    [edgeBoxSamples] = regionSampler.sampleEdgeBoxes(I);
    
    %x2(I); plotBoxes(s(1:100,:))
    
    [samples,labels] = sampleAroundMouth(fra_db(t),regionSampler,false);
    sampleUnion = BoxUnion(samples);
    all_corners = {};
    all_box_inds = {};
    
    for iBox = 1:size(edgeBoxSamples,1)
        all_corners{iBox} = box2Pts(edgeBoxSamples(iBox,:));
        all_box_inds{iBox} = iBox*ones(4,1);
    end
    
    all_corners = cat(1,all_corners{:});
    all_box_inds = cat(1,all_box_inds{:});
    
    Z = box2Region(sampleUnion,size2(I));
    corner_in_region=Z(sub2ind2(size2(I),fliplr(all_corners)));
    
    goodBoxes = unique(all_box_inds(corner_in_region));
    figure(1)
    
    for iBox = 1:length(goodBoxes)
        clf; imagesc2(I); plotBoxes(edgeBoxSamples(goodBoxes(iBox),:));
        dpc(.1)
    end
    pts2Box(edgeBoxSamples(1,:))
    
    % intersect edge boxes and points of interaction
    [ovps,ints] = boxesOverlap(edgeBoxSamples,samples);
    % sample box has to be inside edge-box and near its edge
    mouthCorners = fra_db(t).face_landmarks.xy(4:5,:);
    [Z,allPts] = paintLines(zeros(size2(I)),[mouthCorners(1,:) mouthCorners(2,:)]);
    Z = bwdist(Z) < size(I,1)/6;
    x2(Z); x2(I);
    
    
    %[rr,cc] = LineTwoPnts(mouthCorners(1,1),mouthCorners(1,2),mouthCorners(2,1),mouthCorners(2,1));
    
    %     x2(I);
    %     plot(rr,cc,'r-');
    
    
    ff = find(sum(ovps,2)>0);
    
    for itt = 1:length(ff)
        tt=ff(itt)
        clf; imagesc2(I); plotBoxes(samples);
        plotBoxes(edgeBoxSamples(tt,:),'r-','LineWidth',2);
        dpc
    end
    
    
    
    
    
    if doDebugging
        
        clf; imagesc2(I);
        plotBoxes(samples(labels==0,:),'g-')
        plotBoxes(samples(labels==1,:),'r-');
        dpc
    end
    
    pos_interaction_samples{end+1} = multiCrop2(I,samples(labels==1,:));
    neg_interaction_samples{end+1} = multiCrop2(I,edgeBoxSamples(labels==0,:));
    %
end
%%
train_neg_f = find(train_neg);
nNegPerImage = 2;
train_neg_f = vl_colsubset(train_neg_f,inf,'Uniform');
[negRegionData] = collectRegionsAndFeatures(conf,fra_db(train_neg_f),false,nNegPerImage,featureExtractor);
%% train classifier to distinguish between regions
pos_feats = [posRegionData.feats];
neg_feats = [negRegionData.feats];
% [posFeats,negFeats] = splitFeats(all_features,all_labels)
[x,y] = featsToLabels(pos_feats,neg_feats);
% x = vl_homkermap(x,1);
[w b info] = vl_svmtrain(x, y, .001);
%x2({fra_db(~isValid).I});
% try on a few images...
%%
test_pos_f = find(isClass & ~isTrain & isValid);
test_neg_f = find(~isClass & ~isTrain & isValid);
curSet = test_pos_f;
for t = 8:length(curSet)
    iImage = curSet(t);
    imgData = fra_db(iImage);
    regionData = collectRegionsAndFeatures(conf,imgData,false,inf,featureExtractor);
    if ~regionData(0).valid
        warning('invalid image data - skipping');
        continue
    end
    regions = {regionData.region};
    feats = [regionData.feats];
    %     feats = vl_homkermap(feats,1);
    scores = w'*feats;
    %         displayRegions(imgData.I,regions,scores,0,5);
    %         continue
    %     w'*[regionData.feats]
    %     z = zeros(size2(imgData.I));
    z = -100*ones(size2(imgData.I));
    z_mask = false(size(z));
    for iRegion = 1:length(regions)
        curRegion = regions{iRegion};
        %         curRegion = imdilate(curRegion,ones(3));
        z(curRegion) = max(z(curRegion),scores(iRegion));
        z_mask(curRegion) = true;
        %                 z = z+scores(iRegion)*curRegion;
    end
    z(~z_mask) = min(z(z_mask));
    z = normalise(double(z));
    clf;imagesc2(sc(cat(3,z,im2double(imgData.I)),'prob_jet'));
    dpc
    %     cat(4,regionData.region)
end
%%
net = init_nn_network;
net.layers = net.layers(1:16);
all_feats = extractDNNFeats({fra_db.I},net,17);
% all_feats_pos = extractDNNFeats({fra_db(train_pos).I},net,16);
% all_feats_neg = extractDNNFeats({fra_db(train_neg).I},net,16);
%%

posClass = 5; % brusing teeth
isClass = [fra_db.classID] == posClass;
train_pos = isClass & isTrain & isValid;
train_neg = ~isClass & isTrain & isValid;
ttt = 1;
% pos_feats = all_feats_pos(ttt).x;
% neg_feats = all_feats_neg(ttt).x;
pos_feats = all_feats.x(:,train_pos);
neg_feats = all_feats.x(:,train_neg);
% [posFeats,negFeats] = splitFeats(all_features,all_labels)
[x,y] = featsToLabels(pos_feats,neg_feats);
[w b info] = vl_svmtrain(x, y, .0001);

vl_pr(2*(isClass(~isTrain & isValid))-1,w'*all_feats.x(:,~isTrain & isValid))

%%
test_pos_f = find(isClass & ~isTrain & isValid);
cur_subset = test_pos_f;

figure(1);
for it = 1:length(cur_subset)
    t = cur_subset(it);
    [I,R] = getRegionImportance(conf,fra_db(t).I, [w;b], net,[15]);
    clf; imagesc2(R);
    dpc;
end
%F = featureExtractor.extractFeatures({fra_db.I});
%% some visualizations:
% 1. show the quality of best IOU score by using edgeboxes
all_ovps = zeros(size(f_train_pos));

for iu = 1:length(f_train_pos)
    t = f_train_pos(iu);
    I = fra_db(t).I;
%     curbb = pts2Box(fra_db(t).\);
    s = size2(fra_db(t).I);
    curbb = BoxIntersection(curbb,[1 1 fliplr(s)]);
    curBoxes = edgeBoxes(I,model,opts);
    curBoxes(:,3:4)=curBoxes(:,3:4)+curBoxes(:,1:2);
    [ovps,ints] = boxesOverlap(curBoxes,curbb);
    all_ovps(iu) = max(ovps);
    clf; plot(all_ovps);
    drawnow; %pause(.01);
end
%figure,hist(all_ovps,10)
showSorted({fra_db(f_train_pos).I},all_ovps);

%% now show result sorted from worst to best, at image level, too
close all
[r,ir] = sort(all_ovps,'ascend');
f_train_pos_s = f_train_pos(ir);
for iu = 1:length(f_train_pos)
    iu
    t = f_train_pos_s(iu);
    I = fra_db(t).I;
    
    curbb = pts2Box(fra_db(t).gt_obj);
    s = size2(fra_db(t).I);
    curbb = BoxIntersection(curbb,[1 1 fliplr(s)]);
    curBoxes = edgeBoxes(I,model,opts);
    curBoxes(:,3:4)=curBoxes(:,3:4)+curBoxes(:,1:2);
    [ovps,ints] = boxesOverlap(curBoxes,curbb);
    [b,ib] = sort(ovps,'descend');
    %
    clf; subplot(1,2,1); imagesc2(I);
    plotBoxes(curbb,'g-','LineWidth',2);
    plotBoxes(curBoxes(ib(1),:),'m--','LineWidth',2);
    title(num2str(b(1)));
    
    I_orig = getImage(conf,fra_db(t));
    curBoxes = edgeBoxes(I_orig,model,opts);
    curBoxes(:,3:4)=curBoxes(:,3:4)+curBoxes(:,1:2);
    ff = fra_db(findImageIndex(fra_db,fra_db(t).imageID));
    curbb = pts2Box(ff.objects.poly);
    %     curbb = ff.hands;
    if isempty(curbb),continue,end
    %     curbb = ff.obj_bbox;
    [ovps,ints] = boxesOverlap(curBoxes,curbb);
    [b,ib] = sort(ovps,'descend');
    %
    subplot(1,2,2); imagesc2(I_orig);
    plotBoxes(curbb,'g-','LineWidth',2);
    plotBoxes(curBoxes(ib(1),:),'m--','LineWidth',2);
    dpc;
end


