
% 5/7/2015
%---------
% Action Object Detection - a structured output framework for action object detection
% and description.
cd ~/code/mircs
initpath;
config;
%addpath('/home/amirro/code/3rdparty/edgeBoxes/');
addpath('/home/amirro/code/3rdparty/liblinear-1.95/matlab');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/examples/');
addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta11_gpu/matlab/');

% vl_compilenn('enableGpu',true);


%%
addpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained');
install

%%
load fra_db;
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
featureExtractor = DeepFeatureExtractor(conf,true)
%%
%% classify small sub images to prove the point
% collect training data - these are all the image segments
curSet = f_train_pos;
%all_data = struct('box',{},'mask',{},'img',{});
all_data = struct('img',{},'regions',{},'gt_ovp',{});
n = 0;
goods = false(size(curSet));
% training
for it = 1:length(curSet)
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
    [I_sub,mouthBox,candidates] = getCandidateRegions(conf,imgData);
    roiMask = cropper(gt_graph{2}.roiMask,mouthBox);
    if nnz(roiMask)==0
        continue
    end
    regions = [row(candidates.masks) roiMask];
    [ovps ints uns] = regionsOverlap3({roiMask},candidates.masks);
    
    %posSamples = vl_colsubset(candidates.masks(ovps>.5),10);
    %posSamples{end+1} = roiMask;
    %negSamples = vl_colsubset(candidates.masks(ovps<.05),10);
    %displayRegions(I_sub,candidates.masks(ovps==0),ovps(ovps==0),.01)
    n = n+1;
    all_data(n).img = I_sub;
    all_data(n).regions = regions;
    all_data(n).gt_ovp = ovps;
    %     all_data(n).posmasks = posSamples;
    %     all_data(n).negmasks = negSamples;
    goods(it) = true;
end


%%
all_pos_samples = {};
all_neg_samples = {};
all_pos_samples_i = {};
all_neg_samples_i = {};

pos_rel_sizes = {};
pos_abs_sizes= {};
for t = 1:length(all_data)
%     if t==92,continue,end
    t
    curImg = all_data(t).img;                
    pos_rel_sizes{end+1} = cellfun(@(x) nnz(x)/prod(size2(curImg)),all_data(t).posmasks);
    pos_abs_sizes{end+1} = cellfun(@(x) nnz(x),all_data(t).posmasks);
end
hist([pos_rel_sizes{:}])


for t = 1:length(all_data)
%     if t==92,continue,end
    t
    curImg = all_data(t).img;            
    for u = 1:length(all_data(t).posmasks)
        all_pos_samples{end+1} = maskedPatch(curImg,all_data(t).posmasks{u},true,.5);
        all_pos_samples_i{end+1} = t;
    end
    for u = 1:length(all_data(t).negmasks)
        all_neg_samples{end+1} = maskedPatch(curImg,all_data(t).negmasks{u},true,.5);
        all_neg_samples_i{end+1} = t;
    end    
end


sizes = cellfun3(@(x) prod(size2(x)),all_neg_samples)
[m,im] = sort(sizes,'ascend');
all_pos_samples_i = [all_pos_samples_i{:}];
all_neg_samples_i = [all_neg_samples_i{:}];
all_neg_samples(sizes<10) = [];
displayImageSeries2(all_neg_samples(im))

feats_pos_local = featureExtractor.extractFeaturesMulti(all_pos_samples);
feats_neg_local = featureExtractor.extractFeaturesMulti(all_neg_samples);

pos_masks = cat(2,all_data.posmasks);pos_masks(cellfun(@nnz,pos_masks)<64) = [];pos_masks = cellfun2(@(x) repmat(x,[1 1 3]),pos_masks);
neg_masks = cat(2,all_data.negmasks);neg_masks(cellfun(@nnz,neg_masks)<64) = [];neg_masks = cellfun2(@(x) repmat(x,[1 1 3]),neg_masks);
feats_pos_mask = featureExtractor.extractFeaturesMulti(pos_masks);
feats_neg_mask = featureExtractor.extractFeaturesMulti(neg_masks);

[x,y] = featsToLabels(feats_pos_local,feats_neg_local);
p = Pegasos(x,y,'lambda',.01,'foldNum',5,'bias',1);
w = p.w(1:end-1);

[x_mask,y_mask] = featsToLabels(feats_pos_mask,feats_neg_mask);
p_m = Pegasos(x_mask,y_mask,'lambda',.01,'foldNum',5,'bias',1);
w_m = p_m.w(1:end-1);
%%
curSet = f_test_neg;
for it = 17:length(curSet)
    t = curSet(it);
    t
    imgData = fra_db(t);
    I=  getImage(conf,imgData);
    [curImg,mouthBox,candidates] = getCandidateRegions(conf,imgData);
    masks = candidates.masks;
    sizes = cellfun3(@nnz,masks);
    sel_remove = sizes < 64 | sizes > prod(size2(curImg))*.15;
    masks(sel_remove) = [];
        
    s_masks = cellfun2(@(x) repmat(x,[1 1 3]),masks);
    curShapeFeats = featureExtractor.extractFeaturesMulti(s_masks);
    shapeScores = w_m'*curShapeFeats;
    
    %     [r,ir] = sort(shapeScores,'descend');
    %     shapeScores    
    curSamples = {}
    for u = 1:length(masks)
        curSamples{end+1} = maskedPatch(curImg,masks{u},true,.5);
    end
                
    curFeats = featureExtractor.extractFeaturesMulti(curSamples);
    curScores = w'*curFeats;
    
    totalScores = curScores+shapeScores;
    %curScores = w'*curFeats;
%     skippy = true(size(curScores));
%     skippy(1:10:end) = false;
    %     curScores(skippy) = -inf;
    displayRegions(curImg,masks,totalScores,0,5);
end
