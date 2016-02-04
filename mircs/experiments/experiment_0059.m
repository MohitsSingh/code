% Experiment 0057, Feb 2 2015
%------------------------------
% train a deep network to tell faces from non faces.
initpath;
config
addpath(genpath('~/code/utils'));
addpath('/home/amirro/storage/data/VOCdevkit');
addpath('/home/amirro/storage/data/VOCdevkit/VOCcode/');
addpath('/home/amirro/code/3rdparty/edgeBoxes/');
model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
addpath('/home/amirro/code/3rdparty/vlfeat-0.9.18/toolbox');
vl_setup
addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7'));
addpath('~/code/common');
vl_setupnn
% net = init_nn_network('imagenet-vgg-s.mat');
addpath('/home/amirro/code/myFaceDetector');
basePath = '~/storage/mscoco/val2014/COCO_val2014_%012.0f.jpg';
basePath_train = '~/storage/mscoco/train2014/COCO_train2014_%012.0f.jpg';

% 1. obtain face images and non faces images
% train using matconvnet

load ~/storage/misc/aflw_crops.mat % ims scores
figure,hist(scores)
%length(ims(scores>2.5))
pos_ims = ims(scores>2.5);
pos_ims = cellfun2(@(x) imResample(x,[32 32],'bilinear'), pos_ims);

person_ids = dlmread('~/val_person_ids.txt');
non_person_ids = dlmread('~/val_non_person_ids.txt');
non_person_ids_train = dlmread('~/train_non_person_ids.txt');

% get paths of non-person images for false face detections.
person_paths = {};
for t = 1:length(person_ids)
    curPath = sprintf(basePath,person_ids(t));
    person_paths{t} = curPath;
end
non_person_paths = {};
for t = 1:length(non_person_ids)
    curPath = sprintf(basePath,non_person_ids(t));
    non_person_paths{t} = curPath;
end

% calc face boxes for each of the non person paths and store the
% results.....

nBoxesPerImage = 1000;
nImages = length(non_person_paths);
totalBytesBoxes = nImages*nBoxesPerImage*4;

model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 1e3;  % max number of boxes to detect

save non_person_paths_val.mat non_person_paths
non_person_paths_train = {};
for t = 1:length(non_person_ids_train)
    curPath = sprintf(basePath_train,non_person_ids_train(t));
    non_person_paths_train{t} = curPath;
end

all_boxes = struct('imagePath',{},'boxes',{});
%
id = ticStatus('extracting lots and lots of boxes...',.5,.5);
for u = 1:length(non_person_paths)
    
    curPath = non_person_paths{u};
    I = imread2(curPath);
    curBoxes = edgeBoxes(I,model,opts);
    curBoxes = curBoxes(:,1:4);
    curBoxes(:,3:4) = single(curBoxes(:,3:4)+curBoxes(:,1:2));
    all_boxes(u).imagePath = curPath;
    all_boxes(u).boxes = boxes;
%     curSubs = multiCrop2(I,round(curBoxes(:,1:4)));
%     curSubs = cellfun2(@(x) imResample(x,[32 32],'bilinear'), curSubs);
    %x2(I); plotBoxes(curBoxes(1:10,:))
    tocStatus(id,u/length(non_person_paths));
end

save edgeBoxesValidation.mat all_boxes


%% train the net....

