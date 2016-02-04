function res = extract_head_feats(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    addpath('~/code/SGE');
    cd ~/code/mircs;
    initpath;
    config;
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta7/matlab/');
    addpath('/home/amirro/code/common');
    vl_setup;
    vl_setupnn;
    res.net_s = init_nn_network('imagenet-vgg-s.mat');
    res.net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
    res.conf = conf;
    load ~/storage/misc/s40_fra_faces_d_new;
    res.s40_fra_faces_d = s40_fra_faces_d;
    return;
end
% res = struct('type',{},'x',{});
res = struct('result',[]);
% res(5).x = [];
boxes = params.boxes;
if (isempty(boxes(1)) || isnan(boxes(1)) || isinf(boxes(1)))
    return
end
conf = initData.conf;
[I,I_rect] = getImage(conf,params.name);
curFaceBox = params.boxes(1,1:4)+I_rect([1 2 1 2]);

kp_global = params.kp_global;
% % clf; imagesc2(I);
% % plotBoxes(I_rect);
% % plotBoxes(params.kp_global)
% % plotBoxes(curFaceBox);
poseData.faceBox = curFaceBox;
poseData.landmarks = [boxCenters(kp_global(:,1:4)) kp_global(:,end)];
% showCoords(poseData.landmarks,[],'color','y');
res = struct('result',featuresFromHead(I, poseData, initData.net_s,16,initData.net_deep,33,params.partOfHead));
% pause
end