function res = extract_dnn_feats_imgs(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    
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
    %     res.net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
    
    return;
end
% facesDir = params.faceDir;
img = params.img;
%'~/storage/aflw_face_imgs';
% load(fullfile(facesDir,[params.name '.mat']),'img');
res.feats_s = extractDNNFeats(img,initData.net_s,16); % get fc6,fc7,fc8 for both nets.
% res.feats_s = extractDNNFeats(img,initData.net_s,[16 18 20]); % get fc6,fc7,fc8 for both nets.
% res.feats_deep = extractDNNFeats(img,initData.net_deep,[33 35 37]);
