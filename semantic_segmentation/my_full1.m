caffe_root='/home/liav/code/3rd/caffe/'; % caffe root folder
addpath('/home/liav/code/other');
addpath('/home/liav/code/hands/');
addpath(genpath(fullfile(caffe_root,'matlab')));
addpath('/home/liav/code/hands/');
expDir='/net/mraid11/export/data/amirro/my_semantic_segmentation/';
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
dataDir = fullfile(expDir,'data');
addpath(genpath('~/code/utils'));
% mkdir();
mkdir(expDir);
patches_fname=fullfile(expDir,'full_gt_hand_bbox.mat');
gpuID=1; % -1 for CPU (should also change in the solver prototxt file)
caffe_model_to_finetune=fullfile(dataDir,'VGG_ILSVRC_16_layers_conv.caffemodel'); %TODO : copy to ~/storage (from /home
caffe_model=fullfile(dataDir,'fcn-32sfo_iter_50000.caffemodel'); % output model
caffe_solver_file = fullfile(dataDir,'fcn-32sfo_solver.prototxt'); % sent to me by liav
caffe_deploy_conv_file = fullfile(dataDir,'fcn-32sfo_deploy.prototxt');


addpath('/home/amirro/code/3rdparty/vlfeat-0.9.19/toolbox');vl_setup;
NSAMPLES=10;
% conf.USE_GT_HAND=1; %
conf.IGNORE_VALUE=99; % "TODO: this is don't care (defined in the prototxt as well)
% net_input_wh=[448 448];
% %%
% curDir = pwd;
% cd ~/code/mircs;
% initpath;
% cd(curDir);
% my_data = my_get_full_data;
%
% cd ~/code/mircs
% initpath;
% config;
%
% images = {};
% masks = {};
% isTrain = [fra_db.isTrain];
% for t = 1:length(fra_db)
%     t
% %     if ~isTrain(t),continue,end
%     imgData = fra_db(t);
%     I = getImage(conf,imgData);
%     [I_sub,~,mouthBox] = getSubImage2(conf,imgData,true);
%     %[mouthMask,curLandmarks] = getMouthMask(I_sub,mouthBox,imgData.Landmarks_dlib,dlib,imgData.isTrain);
%     [groundTruth,isValid] = getGroundTruthHelper(imgData,params,I,mouthBox);
%     images{t} = I_sub;
%     masks{t} = groundTruth;
% end

% save images_and_masks.mat images masks isTrain
%

%load /net/mraid11/export/data/amirro/my_semantic_segmentation/images_and_masks
load ~/storage/misc/images_and_masks
my_data = {};
net_input_wh = [224 224];
for t = 1:length(images)
    seg_labels = uint8((imResample(double(masks{t}),net_input_wh)) > 0);
    my_data{t} = struct('image',double(im2uint8(imResample(images{t},net_input_wh))),'seg_labels',seg_labels);
end

sel_train = find(isTrain);
sel_val = sel_train(1:3:end);
sel_train = setdiff(sel_train,sel_val);

all_labels = col(cellfun2(@(x) x.seg_labels,my_data));
all_images = col(cellfun2(@(x) x.image,my_data));
cd(fullfile(dataDir,'..'));
%%
write_as_hdf3(fullfile(dataDir,'train_full_fo-face_gt'),all_labels(sel_train),all_images(sel_train));
write_as_hdf3(fullfile(dataDir,'val_full_fo_face_gt'),all_labels(sel_val),all_images(sel_val));

%% write test data
sel_test = find(~isTrain);
write_as_hdf3(fullfile(dataDir,'test_full_fo-face_gt'),all_labels(sel_test),all_images(sel_test));

%% Train Caffe
log_file='c1.log';
fprintft('Start\n');
caffe_train(caffe_root,caffe_solver_file,gpuID,log_file,caffe_model_to_finetune)
fprintft('Done\n');
% system([caffe_root, '/build/tools/caffe train ', sprintf('-gpu %d -solver %s -snapshot %s |& tee c2_.log', gpuID, caffe_solver_file,'data/fcn-32sfo_iter_26100.solverstate')]);
% caffe_test(caffe_root,gpuID,'handseghdf5/fcn-32so-hand_gt_train_val.prototxt','data/fcn-32so-hand_gt_iter_870.caffemodel',150);
%% view
log_file='c1.log';
while 1
    caffe_parse_log(caffe_root,log_file,1);
    pause(30)
end
%% Test - load model
caffe_model=fullfile(dataDir,'fcn-32sfo_iter_8100.caffemodel'); % output model
caffe.reset_all;
caffe.set_mode_gpu;
caffe.set_device(gpuID);
net_conv = caffe.Net(caffe_deploy_conv_file, caffe_model, 'test'); % create net and load weights
%% Test single image
%h5name='data/train_full_fo-face_gt-1.h5';
stored_data = {};
stored_label = {};
for t = 1
    t    
    h5name=['data/test_full_fo-face_gt-' num2str(t) '.h5'];
    %     h5name=['data/val_full_fo_face_gt-' num2str(t) '.h5'];
    stored_data{t}=hdf5read(h5name,'/data');
    stored_label{t}=hdf5read(h5name,'/label');
    %     all_stored_data{t} = stored_data;
    %     all_stored_label{t} = stored_label;
end
stored_data  = cat(4,stored_data{:});
stored_label  = cat(4,stored_label{:});

%%
for ii=1:3:size(stored_data,4)
    %     ii = ii:ii+2;
    extra_str= ''; % '_val'
    ii
    resPath = j2m('/home/amirro/notes/images/2015_10_08',[extra_str fra_db(sel_test(ii)).imageID],'.png');
    %     if exist(resPath,'file'),continue,end
    data_permute=stored_data(:,:,:,ii);
    label_permute=stored_label(:,:,ii);
    label=permute(label_permute,[2 1 3 4]);
    im=permute(data_permute,[2 1 3]);
    mean_pix = [103.939, 116.779, 123.68];
    bgr=im(:,:,[3 2 1],:);
    for c = 1:3
        bgr(:, :, c, :) = bgr(:, :, c, :) + mean_pix(c);
    end
    im=uint8(bgr(:,:,[3 2 1],:));
    % figure(1); clf; imshow(im)
    imsz=size2(im);
    MARGIN_SZ=[0 0];
    downsample=1;
    if (length(size(data_permute))==3)
        net_conv.blobs('data').reshape([size(data_permute) 1]);
    else
        net_conv.blobs('data').reshape(size(data_permute));
    end
    net_conv.blobs('data').set_data(data_permute);
    net_conv.forward_prefilled();
    prob = net_conv.blobs('upsample1').get_data();
    prob=permute(prob,[2 1 3 4]);
    figure(1);
    mm = 1;
    nn = 3;
    %     my_subplot = @vl_tightsubplot;
    gap = .02;
    %clf; tight_subplot(mm,nn,1,gap);
    clf;ha = tight_subplot(mm,nn,[0 0],[0 0],0);
    % figure(9); clf;
    axes(ha(1));
    imshow(label); title('ground truth');
    % GT
    [m,mi]=max(prob,[],3);
    mi=imresize(mi,[size(im,1) size(im,2)],'nearest');
    axes(ha(2));
    %tight_subplot(mm,nn,2,gap);
    imshow(imfuse(im,mi));  title('dnet result');% est
    %subplot(mm,nn,3); imagesc2(mi); %colorbar; %est
    im = im(:,:,[3 2 1]);
    %tight_subplot(mm,nn,3,gap);
    axes(ha(3));
    imshow(im); title('orig');
    %saveas(gcf,j2m('/home/amirro/notes/images/2015_10_08',fra_db(sel_test(ii)),'.png'));
    %     saveas(gcf,resPath)
    dpc
end


%%
%% run on patches
addpath('~/code/3rdparty/sc');
addpath('~/code/mircs');
seg_maps = {};
seg_maps_detected_faces = {};
for u = 1:length(images_detected_face)
    u
%     if ~fra_db(u).isTrain,continue,end
    net_input_wh = [224 224];  
    mean_pix = [103.939, 116.779, 123.68];
    im = images_detected_face{u};
    data_permute = transformData(images_detected_face(u),mean_pix,net_input_wh);
    seg_maps_detected_faces{u} = compute_seg(net_conv,data_permute,im);    
    im = images{u};
    data_permute = transformData(images(u),mean_pix,net_input_wh);
    seg_maps{u} = compute_seg(net_conv,data_permute,im);            
end

save ~/storage/misc/images_and_masks_res.mat seg_maps_detected_faces seg_maps

%%
%run on entire images
addpath('~/code/mircs');
initpath;config;
seg_maps = {};
addpath('~/code/semantic_segmentation/');
%%
for u = 1100:1:length(images_detected_face)
    u
    if ~fra_db(u).isTrain,continue,end
    net_input_wh = [224 224];  
    mean_pix = [103.939, 116.779, 123.68];
    %im = images_detected_face{u};
    conf.get_full_image = true;
    im = getImage(conf,fra_db(u));
    im = cropper(im,round(inflatebbox(fra_db(u).faceBox,2,'both',false)));
    data_permute = transformData({im},mean_pix,net_input_wh);
    curSegMap = compute_seg(net_conv,data_permute,im);        
    clf; subplot(1,2,1); imagesc2(im);
    subplot(1,2,2); imagesc2(sc(cat(3,curSegMap,im),'prob'));
    dpc;
end

%


%% Test original images
mean_pix = [103.939, 116.779, 123.68];
% im=...


% im = images{1};

data_permute=single(im(:,:,[3 2 1],:));
for c = 1:3
    data_permute(:, :, c, :) = data_permute(:, :, c, :) - mean_pix(c);
end
data_permute=permute(data_permute,[2 1 3 4]);
% figure(1); clf; imshow(im)
downsample=1;
data_permute=imresize(data_permute,[448 448]);
net_conv.blobs('data').reshape([size(data_permute) 1]);
net_conv.blobs('data').set_data(data_permute);
net_conv.forward_prefilled();

prob = net_conv.blobs('upsample1').get_data();
prob=permute(prob,[2 1 3 4]);
[m,mi]=max(prob,[],3);
mi=imresize(mi,[size(im,1) size(im,2)],'nearest');
figure(5); clf;imshow(imfuse(im,mi))
figure(4); clf;imagesc(mi); colorbar;



%%
% d = getAllFiles('~/notes/images/2015_10_08')
% 
% nn = batchify(length(d),length(d)/6);
% 
% for t = 1:length(nn)
%     curBatch = d(nn{t});
%     imgs = cellfun2(@(x) imread(x),curBatch);
%     for tt = 1:length(imgs)
%         imgs{tt}(:,end-3:end,1)
%         
%         
        
