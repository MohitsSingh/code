% function fcnTrain_person_parts(varargin)
%FNCTRAIN Train FCN model using MatConvNet

run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('/home/amirro/storage/pascal_parts/');
addpath(genpath('~/code/utils'));
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s_person_parts_focused_sym' ;
opts.dataDir = '/net/mraid11/export/data/amirro//fcn/data/fcn32s-action_obj' ;
mkdir(opts.expDir);
mkdir(opts.dataDir);
opts.modelType = 'fcn8s' ;
%opts.sourceModelPath = '/net/mraid11/export/data/amirro//fcn/data/models/imagenet-vgg-verydeep-16.mat' ;
opts.sourceModelPath = '/net/mraid11/export/data/amirro//matconv_data/imagenet-vgg-verydeep-16.mat';
% [opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup

% setup pascal parts

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat') ;

if exist(opts.imdbPath,'file')
    load(opts.imdbPath);
else
    cmap = VOClabelcolormap();
    pimap = part2ind();     % part index mapping
    
    % create lookup table...
    personmap = pimap{15};
    newMap = containers.Map;
    person_keys = personmap.keys;
    LUT = zeros(size(person_keys));
    newPersonMap = containers.Map;
    for t = 1:length(person_keys)
        curKey = person_keys{t};
        rightKey = ['r' curKey(2:end)];
        f = strmatch(rightKey,person_keys);
        if curKey(1)~='l' || isempty(f) % either does not start with left or no matching right key,
            % leave label as is (map to self)
            fprintf('current key is %s, leaving as is (value is %d)\n',curKey,personmap(curKey))
            LUT(personmap(curKey)) = personmap(curKey);
        else % found a matching "rightkey" --> map current to value of right key
            %
            fprintf('current key is %s, mapping to value of right key, which is (value is %d)\n',curKey,personmap(rightKey))
            LUT(personmap(curKey)) = personmap(rightKey);
        end
    end
    
    % phew...now find the unique values
    U = unique(LUT);        
    oldToNewLUT = containers.Map;
    for ii = 1:length(U)
        % find which old values pointed to such a key and point them to a
        % new key                
        LUT(LUT==U(ii)) = ii;
    end
    
    %%
            
    imagesPath = '/home/amirro/storage/fcn/';
    annotationsPath = '~/storage/pascal_parts/Annotations_Part';
    imgsPath = '/home/amirro/storage/fcn/data/voc11/JPEGImages';
    all_annos = dir(fullfile(annotationsPath,'*.mat'));
    all_annos = all_annos(1:1500); %small...
    imagePaths = {};
    annoPaths = fullfile(annotationsPath,{all_annos.name});
    %     images = {};
    %     labels = {};
    
    images_data = {};
    labels = {};
    
    M = 0;
    for t = 1:length(all_annos)
        %         t
        if (mod(t,50)==0)
            disp(t/length(all_annos));
        end
        load(annoPaths{t});
        if none(ismember(15,[anno.objects.class_ind])),
            continue
        end
        imgPath = fullfile(imgsPath,[all_annos(t).name(1:end-4) '.jpg']);
        %         images{end+1} = imgPath;
        %         labels{end+1} = annoPaths{t};%uint8(single(sel_).*single(part_mask));
        %
        
        % find all person instances.
        imgPath = fullfile(imgsPath,[all_annos(t).name(1:end-4) '.jpg']);
        img = vl_imreadjpeg({imgPath});
        img = img{1};
        [cls_mask, inst_mask, part_mask] = mat2map(anno, img, pimap);
        person_instances = single(inst_mask).*single(cls_mask==15);
        person_instances = RemapLabels(person_instances)-1;
        personRegions = regionprops(person_instances,'BoundingBox','Area','PixelIdxList');
        personRegions([personRegions.Area]<2000) = [];
        
        if length(personRegions)==0
            t
            continue
        end
        
        part_mask = uint8(single(part_mask).*single(cls_mask==15));
        part_mask(part_mask>24) = 0;
        
        %         continue
        for iRegion = 1:length(personRegions)
            %             z = false(size2(img));
            %             z(personRegions(iRegion).PixelIdxList) = 1;
            
            bb = personRegions(iRegion).BoundingBox;
            bb(3:4) = bb(3:4)+bb(1:2);
            bb = round(inflatebbox(bb,1.2,'both',false));
            %             clf; imagesc2(img/255);
            %             plotBoxes(bb);
            
            
            curSubImg = uint8(cropper(img,bb));
            curLabels = cropper(part_mask,bb);
            
            
            curLabels = cropper(part_mask,bb);
            %         x2(curLabels)
            newVals = LUT(curLabels(curLabels>0));
            newLabels = zeros(size(curLabels));
            newLabels(curLabels(:)>0) = newVals;
            curLabels = newLabels;
            
            clf; subplot(1,2,1); imagesc2(curSubImg);
            subplot(1,2,2); imagesc2(curLabels);
            dpc;
            images_data{end+1} = curSubImg;
            labels{end+1} = curLabels;
        end
    end
    %%
    imdb.train = 1:500;%:length(all_annos);
    imdb.val = imdb.train(1:3:end);
    train = setdiff(imdb.train,imdb.val);
    val = imdb.val;
    imdb.test = max([imdb.train imdb.val])+1:length(images_data);
    %     imdb.image_paths = images;
    if (isfield(imdb,'image_paths'))
        imdb = rmfield(imdb,'image_paths');
    end
    imdb.images = 1:length(images);
    imdb.images_data = images_data;
    imdb.labels = labels;
    imdb.nClasses = 24;
    save(opts.imdbPath,'imdb');
end

opts.numFetchThreads = 1 ; % not used yet
% training options (SGD)
opts.train.batchSize = 20;
%opts.train.numSubBatches = 5;
opts.train.numSubBatches = 5;
opts.train.continue = true ;
opts.train.gpus = 1;%[1 2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 0.0001 * ones(1,50) ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

% opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from VGG-VD-16
opts.nClasses = imdb.nClasses;
net = fcnInitializeModel_action_obj('sourceModelPath', opts.sourceModelPath,'nClasses',imdb.nClasses) ;
if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s'}))
    % upgrade model to FCN16s
    net = fcnInitializeModel16s(net) ;
end
if strcmp(opts.modelType, 'fcn8s')
    % upgrade model fto FCN8s
    net = fcnInitializeModel8s(net) ;
end
stats.rgbMean =[116.6725  111.5917  103.1466]';
net.meta.normalization.rgbMean = stats.rgbMean;
%
% net.meta.classes = {'face','object'};

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 1 ;
bopts.labelOffset = 1 ;
bopts.classWeights = ones(1,imdb.nClasses+1,'single') ; % AMIR - was 21.
bopts.rgbMean = stats.rgbMean ;
bopts.useGpu = numel(opts.train.gpus) > 0 ;
getBatchWrapper = @(x) @(imdb,batch) getBatch_action_obj(imdb,batch,x,'prefetch',nargout==0) ;

% Launch SGD
info = cnn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', imdb.train, ...
    'val', imdb.val) ;

% % -------------------------------------------------------------------------
% function fn = getBatchWrapper(opts)
% % -------------------------------------------------------------------------
% fn = @(imdb,batch) getBatch_parts(imdb,batch,opts,'prefetch',nargout==0) ;
