function res = extract_dnn_feats_imgs_2_faces(initData,params)
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
    res.conf = conf;
    load ~/storage/misc/s40_fra_faces_d_new;
    res.s40_fra_faces_d = s40_fra_faces_d;
    %     res.net_deep = init_nn_network('imagenet-vgg-verydeep-16.mat');
    
    return;
end

net = initData.net_s;
% net = net(1:end-1); % we don't need the last layer...
conf = initData.conf;
conf.get_full_image = true;
[I_full,I_rect] = getImage(conf,params.name);
s40_fra = initData.s40_fra_faces_d;
imgData = s40_fra(findImageIndex(s40_fra,params.name));
I_face = cropper(I_full,round(imgData.faceBox));
I_face_extended = cropper(I_full,round(inflatebbox(imgData.faceBox,2,'both',false)));
res.feats_face = extractDNNFeats(I_face,net,[16 18]); % fc6, fc7
res.feats_face_ext = extractDNNFeats(I_face_extended,net,[16 18]); % fc6, fc7
