% function my_fcnTest(varargin)
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;

% experiment and data paths
opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-voc11' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/voc11' ;

%opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes/net-epoch-100.mat';
opts.modelPath = '/home/amirro/storage/fcn/data/fcn8s-action_obj_8_x2_classes_full/net-epoch-100.mat';
%%
opts.modelFamily = 'matconvnet' ;
opts.nClasses = 0;
% [opts, varargin] = vl_argparse(opts, varargin) ;
% experiment setup
% load ~/code/mircs/images_and_face_obj.mat;
load ~/code/mircs/images_and_face_obj_full.mat;
% load ~/storage/misc/images_and_masks_x2_w_hands.mat

train= find(isTrain);
test = find(~isTrain);
val = train(1:3:end);
train = setdiff(train,val);

imdb.images = 1:length(images);
imdb.images_data = images;
needToConvert = isa(images{1},'double') && max(images{1}(:))<=1;
if needToConvert
    for t = 1:length(images)
        imdb.images_data{t} = im2uint8(imdb.images_data{t});
    end
end
imdb.labels = masks;

% opts = vl_argparse(opts, varargin) ;
opts.useGpu = 0;
% -------------------------------------------------------------------------

% ------------------------------------------------------------------------
[net,imageNeedsToBeMultiple,inputVar,predVar] = my_load_net(opts.modelPath,opts.modelFamily);

% % -------------------------------------------------------------------------
% % Train
% -------------------------------------------------------------------------
imdb.nClasses = 7;
numGpus = 0 ;
nClasses = imdb.nClasses+1;
confusion = zeros(nClasses) ;
net.move('gpu');

labels = {'none','face','hand','drink','smoke','blow','brush','phone'};

%%

% imgDir = '/home/amirro/storage/data/Stanford40/JPEGImages';
imgPaths = getAllFiles(imgDir,'.jpg');
%d = dir('~/storage/data/Stanford40/JPEGImages/*.jpg');
%%
for i = 1:30:numel(imgPaths)
%     te
    imId = imgPaths(i) ;
%     rgb = imResample(single(imdb.images_data{imId}),2,'bilinear');
%     [pred,scores_] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,rgb,1,true);
%     dpc;continue%             
%     y = getBatch_action_obj(imdb, imId,'labelOffset',1,'useFlipping',false,'doScaling',false);    
    %rgb = vl_imreadjpeg( {fullfile(imgDir,fra_db(imId).imageID)});
    rgb = vl_imreadjpeg( imgPaths(i));
    rgb = rgb{1};
    %rgb = imResample(single(imdb.images_data{imId}),[384 384],'bilinear');
    rgb = imResample(rgb,[384 384],'bilinear');
    [pred_full,scores_full] = predict_and_show(net,imageNeedsToBeMultiple,inputVar,rgb,1,true,labels);
    dpc; continue;
    rgb = y{2};    
    anno = y{4};
    lb = single(anno);
        
    % Subtract the mean (color)
    im = bsxfun(@minus, single(rgb), net.meta.normalization.averageImage) ;            
    % Soome networks requires the image to be a multiple of 32 pixels
    if imageNeedsToBeMultiple
        sz = [size(im,1), size(im,2)] ;
        sz_ = round(sz / 32)*32 ;
        im_ = imresize(im, sz_) ;
    else
        im_ = im ;
    end
    
    net.eval({inputVar, gpuArray(im_)}) ;    
    scores_ = gather(net.vars(net.getVarIndex('prediction')).value);    
    [~,pred_] = max(scores_,[],3) ;
    
    if imageNeedsToBeMultiple
        pred = imresize(pred_, sz, 'method', 'nearest') ;
    else
        pred = pred_ ;
    end
    
    % Accumulate errors
    ok = lb > 0 ;
    confusion = confusion + accumarray([lb(ok),pred(ok)],1,[nClasses nClasses]) ;
    
    % Plots
    if mod(i - 1,1) == 0 || i == numel(val)
        [iu, miu, pacc, macc] = getAccuracies(confusion) ;
        fprintf('IU ') ;
        fprintf('%4.1f ', 100 * iu) ;
        fprintf('\n meanIU: %5.2f pixelAcc: %5.2f, meanAcc: %5.2f\n', ...
            100*miu, 100*pacc, 100*macc) ;        
        figure(1) ;
        clf;
        mm  = 1;
        nn = 2;
        subplot(mm,nn,1);
        imagesc2(rgb/255);
        axis image ;       
        
        % Print segmentation
        subplot(mm,nn,2);
        pred(end)=8;
        
        figure(3)
        I1 = rgb/255;
%         clf;imagesc2(I1)        
        cmap_hsv = rgb2hsv(jet(8));
        I1_hsv = rgb2hsv(I1);
        
        h = I1_hsv(:,:,1);
        s = I1_hsv(:,:,2);
        v = I1_hsv(:,:,3);
%         for iPred=2
        for iPred = 2:length(labels)
%             iPred=2;            
            p = pred==iPred;
            h(p) = cmap_hsv(iPred,1);
            s(p) = 1;
        end
        I1_hsv = cat(3,h,s,v);
        I1 = hsv2rgb(I1_hsv);
        clf; imagesc2(I1); 
        
        
        figure(1)
        imagesc(uint8(pred-1)) ;% 
        axis equal        
        colormap(jet(8));
        lcolorbar(labels);
        dpc; continue
        title('predicted') ;
        mm = 2;nn = 4;
        figure(2); clf;
        for u = 1:8            
            subplot(mm,nn,u);
            imagesc2(sc(cat(3,scores_(:,:,u),rgb/255),'prob'));title(labels{u});            
        end
        drawnow ;        
        pause
        continue
        %SZ = 513;
        SZ = size(rgb,1);
        rgb_p = rgb/255;
        rgb_p = imresize(rgb_p,[SZ,SZ],'bilinear');
        imwrite(rgb_p,'d1/1.ppm');
        data = zeros([SZ,SZ,3],'single');
        %data = -inf([SZ,SZ,3],'single');
        data(:,:,1:size(scores_,3)) = imResample(single(scores_),[SZ SZ]);
        SS = normalise(scores_(:,:,2));
        data(:,:,2) = imResample(SS,[SZ SZ]);
        save('feats/1_blob_0.mat','data','-v6');
           
    end
end



