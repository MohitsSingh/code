% experiment 0057: make a foreground model for faces 

initpath
config
% load ~/storage/misc/pos_and_neg_faces.mat
% x2(pos_ims{1})
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

% step 1: get foreground masks for each face(actually, for each human)
% step 2: learn a local model for foreground/background
baseDir = '~/storage/mscoco/val2014';
inputPath = '~/storage/mscoco/val_people.txt';

[ paths,bbs ] = load_coco_bbs(baseDir,inputPath);

% for each image, load the face detection, intersect with bounding boxes
[d,id] = sort(paths);
bbs_1 = bbs(id,:);

% m = cat(1,d{:});
% groupStarts = find(any(diff(m),2));
% if (groupStarts(end)<size(m,1))
%     groupStarts = [groupStarts;size(m,1)+1];
% end
% 
% t = 1;
% for p = 1:length(groupStarts-1)    
%     disp(m(t:groupStarts(p),:))
%     disp('***')
%     t = groupStarts(p)+1;
%     pause
% end

% strmatch(p{1},paths)
% curGroup = 1;
% nextGroup = groupStarts(1);
p = unique(paths);
% groupStarts
% while curGroup < length(groupStarts)
t = 552
%%
while (true)
%     for t = 501:length(p)
    t = t+1
    inds = strmatch(p{t},paths);
    personBoxes = bbs(inds,:);    
    [pathstr,name,ext] = fileparts(p{t});
    faceDets = load(fullfile('~/storage/mscoco/val_faces',[name '.mat']),'detections');
    faceBoxes = faceDets.detections.boxes;    
    if (any(faceBoxes))
%         all_boxes{end+1} = faceBoxes;
        I = imread2(fullfile('~/storage/mscoco/val2014',p{t}));
        clf;imagesc2(I);
        plotBoxes(faceBoxes);
        plotBoxes(personBoxes,'m-','LineWidth',2);
        drawnow
        pause
%         u = multiCrop2(I,round(faceBoxes(:,1:4)));
%         false_ims{end+1} = cellfun2(@(x) imResample(x,[32 32],'bilinear'),u); 
    end
end
x2(I)
%%

% I = imread('~/storage/data/Stanford40/JPEGImages/smoking_111.jpg');
opts = edgeBoxes;
opts.alpha = .65;     % step size of sliding window search
opts.beta  = .75;     % nms threshold for object proposals
opts.minScore = .01;  % min score of boxes to detect
opts.maxBoxes = 5000;  % max number of boxes to detect
opts.minBoxArea = 500;
% bbs = edgeBoxes2(I,model,opts);
tic, bbs_img=edgeBoxes(I,model,opts); toc
bbs_img(:,3) = bbs_img(:,3)+bbs_img(:,1);
bbs_img(:,4) = bbs_img(:,4)+bbs_img(:,2);
[a,b] = BoxSize(bbs_img);
z = a./b <= 1.5 & a./b >= 1/1.5;
bbs_img = bbs_img(z,:);
bbs_img = round(inflatebbox(bbs_img,[1.2 1.2],'both',false));
% x2(I); plotBoxes(bbs_img(1:50,:))
% x2(I),m = getSingleRect(1)
% ovps = boxesOverlap(m,bbs);
% [u,iu] = sort(ovps,'descend')
% x2(I);plotBoxes(bbs(iu(1:5),:))
mm = multiCrop2(I,bbs_img);
mm = cellfun2(@(x) imResample(x,[32 32],'bilinear'),mm);
% mImage(mm);
%
% imgSize = 256;
% opts.maxImageSize = imgSize;
% spSize = 50;
% opts.pixNumInSP = spSize;
% opts.show  =false;     
% [sal,sal_bd,resizeRatio] = extractSaliencyMap(im2uint8(I),opts);
% x2(I)
% x2(sal)
%

load faces_32_data_mean
% load ~/storage/misc/pos_and_neg_faces_16_data_mean.mat
% L = load('/home/amirro/code/mircs/data/faces-32/net-epoch-20.mat');
%L = load('/home/amirro/code/mircs/data/faces-32_big/net-epoch-20.mat');
L = load('/home/amirro/code/mircs/data/faces-32_cifar_f2/net-epoch-20.mat');
net = L.net;
% net = cnn_faces_32;
net.layers = net.layers(1:end-1);
net.normalization.averageImage = dataMean;
net.normalization.border = [0 0];
net.normalization.imageSize = [32 32];
net.normalization.keepAspect = true;
[res] = extractDNNFeats(mm,net,12)
[v,iv] = sort(res.x(1,:),'descend');
bb_v = [bbs_img(:,1:4),res.x(1,:)'];
pick = nms(bb_v, .3);
bb_v = bb_v(pick,:);
mImage(mm(pick));title('net');
% figure,plot(v)
% 
% [map,counts] = computeHeatMap(I,bb_v,'max');
% x2(map);x2(I)

% 
% mImage(mm);

% 
% [w h] = BoxSize(bbs);
% 
% [b,ib] = sort(-w);
% for iu = 1:length(paths)
%     u = ib(iu);
%     I = imread(fullfile(baseDir,paths{u}));
%     curBB = bbs(u,:);
% %     curBB(3:4) = curBB(3:4)+curBB(1:2);
%     clf; imagesc2(I); plotBoxes(curBB);
%     drawnow
%     pause
% end
% 

%figure,hist((w.*h).^.5,100)
%%
I = imcrop(I);
x2(I)
% addpath(genpath('/home/amirro/code/3rdparty/MCG-PreTrained/MCG-PreTrained'));install
[candidates, ucm] = im2mcg(I,'fast',true); % warning, using fast segmentation...

x2(ucm(1:2:end,1:2:end));
x2(I)
displayRegions(I,candidates.masks)

U = imageStackToCell(candidates.masks);

[regions,ovp,sel_] = chooseRegion(I, U,.3);
displayRegions(I,regions,ovp)
