function res = extract_dnn_feats_imgs_2(initData,params)
if (~isstruct(initData) && strcmp(initData,'init'))
    cd ~/code/mircs;
    initpath;
    config;
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta10/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta10/examples/');
    addpath('/home/amirro/code/3rdparty/matconvnet-1.0-beta10/matlab/');
    addpath('/home/amirro/code/common');
    vl_setup;
    vl_setupnn;
    res.net_s = init_nn_network('imagenet-vgg-s.mat');
    res.conf = conf;
    res.net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
    
    return;
end

net = initData.net_s;
% net = net(1:end-1); % we don't need the last layer...
conf = initData.conf;
conf.get_full_image = true;
[I_full,I_rect] = getImage(conf,params.name);
I_crop = cropper(I_full,I_rect);
res.feats_full = extractDNNFeats(I_full,net,[16]); %[16 18]: fc6, fc7
res.feats_crop = extractDNNFeats(I_crop,net,[16]); 
res.feats_full_deep = extractDNNFeats(I_full,initData.net_deep,[33]); %[33 35 38]: fc6, fc7, output
res.feats_crop_deep = extractDNNFeats(I_crop,initData.net_deep,[33]);
res.feats_full_tiled = extractDNNFeats_tiled(I_full,net,[3 3],[16],false); %[16 18]: fc6, fc7
res.feats_crop_tiled = extractDNNFeats_tiled(I_crop,net,[3 3],[16],false);
res.feats_full_deep_tiled = extractDNNFeats_tiled(I_full,initData.net_deep,[3 3],[33],false); %[33 35 38]: fc6, fc7, output
res.feats_crop_deep_tiled = extractDNNFeats_tiled(I_crop,initData.net_deep,[3 3],[33],false);