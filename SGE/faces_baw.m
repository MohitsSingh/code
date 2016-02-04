function res = faces_baw(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd /home/amirro/code/3rdparty/voc-release5
    startup
    load ~/code/3rdparty/dpm_baseline.mat
    res.model = model;    
    % while we're at it, collect the fc6 features too
    addpath(genpath('~/code/utils'));
    addpath('/home/amirro/storage/data/VOCdevkit');
    addpath('/home/amirro/storage/data/VOCdevkit/VOCcode/');
    %     addpath('/home/amirro/code/3rdparty/edgeBoxes/');
    %     model=load('/home/amirro/code/3rdparty/edgeBoxes/models/forest/modelBsds'); model=model.model;
    addpath(genpath('~/code/3rdparty/piotr_toolbox/'));
    addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');    
    vl_setup
    %     addpath(genpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7'));
    %     addpath('~/code/common');
    %     vl_setupnn
    %     res.net = init_nn_network('imagenet-vgg-s.mat');
    
    return;
end

model = initData.model;
detections = struct('rot',{},'boxes',{});
I_orig = imread2(params.path);

resizeFactor = 480.0/size(I_orig,1);
I_orig = imresize(I_orig,resizeFactor,'bilinear');
% if size(I_orig,1) > 1000
%     
% elseif size(I_orig,1) > 500
%     resizeFactor = 1;
% else    
%     resizeFactor = 2;
% end

% rots = -30:10:30;
rots = 0;
for iRot = 1:length(rots)
    I = imrotate(I_orig,rots(iRot),'bilinear','crop');    
    [ds, bs] = imgdetect(I, model,-1);
    top = nms(ds, 0.1);
    if (isempty(top))
        boxes = -inf(1,5);
    end
    detections(iRot).rot = rots(iRot);
    detections(iRot).boxes = ds(top(1:min(5,length(top))),:);
    if (~isempty(detections(iRot).boxes))
        detections(iRot).boxes(:,1:4) = detections(iRot).boxes(:,1:4)/resizeFactor;
    end
end
res.detections = detections;