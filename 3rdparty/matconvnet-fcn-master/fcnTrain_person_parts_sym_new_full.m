% function fcnTrain_person_parts(varargin)
%FNCTRAIN Train FCN model using MatConvNet
addpath(genpath('~/code/utils'));
addpath(genpath('~/code/3rdparty/piotr_toolbox'));
addpath('~/code/3rdparty/sc');
addpath('~/code/3rdparty/export_fig');
addpath('utils/');
addpath ~/code/3rdparty/vlfeat-0.9.19/toolbox/
vl_setup
run matconvnet/matlab/vl_setupnn ;
addpath matconvnet/examples ;
% experiment and data paths

opts.expDir = '/net/mraid11/export/data/amirro//fcn/data/fcn8s_person_parts_sym_new_full' ;
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

if false && exist(opts.imdbPath,'file')
    load(opts.imdbPath);
else
    cmap = VOClabelcolormap();
    pimap = part2ind();     % part index mapping
    
    % create lookup table...
    personmap = pimap{15};
    newMap = containers.Map;
    person_keys = personmap.keys;
    LUT = zeros(size(person_keys));
    %     newPersonMap = containers.Map;
    
    for t = 1:length(person_keys)
        curKey = person_keys{t}; % left key?
        rightKey = ['r' curKey(2:end)];
        f = strmatch(rightKey,person_keys);
        if curKey(1)~='l' || isempty(f) % either does not start with left or no matching right key,
            % leave label as is (map to self)
            fprintf('current key is %s, leaving as is (value is %d)\n',curKey,personmap(curKey))
            LUT(personmap(curKey)) = personmap(curKey);
            %             newPersonMap(curKey) = curKey;
        else % found a matching "rightkey" --> map current to value of right key
            %
            fprintf('current key is %s, mapping to value of right key, which is (value is %d)\n',curKey,personmap(rightKey))
            LUT(personmap(curKey)) = personmap(rightKey);
            %             newPersonMap(curKey) = rightKey;
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
    imgsPath = '/net/mraid11/export/data/amirro/data/VOCdevkit/VOC2012/JPEGImages';
    all_annos = dir(fullfile(annotationsPath,'*.mat'));
    %all_annos = all_annos(1:1500); %small...
    imagePaths = {};
    annoPaths = fullfile(annotationsPath,{all_annos.name});
    %     images = {};
    %     labels = {};
    
    images_data = {};
    labels = {};
    
    
    newClassNames = {'none','head','eyye','ear','brow','nose','mouth','hair','torso',...
        'neck','lower_arm','upper_arm','hand','lower_leg','upper_leg','foot'};
    %
    %     newClassNames = {};
    %     for ii = 1:length(unique(LUT))
    %
    %         posInLut = find(LUT==ii,1,'first');
    %         newClassNames{end+1} = person_keys{ii}
    %
    % %         newClassNames{end+1} = person_keys{find(LUT==ii,1,'first')};
    %     end
    
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
        curLabels = part_mask;
        % %             %         x2(curLabels)
        newVals = LUT(curLabels(curLabels>0));
        newLabels = zeros(size(curLabels));
        newLabels(curLabels(:)>0) = newVals;
        curLabels = newLabels;
        images_data{end+1} = uint8(img);
        labels{end+1} = curLabels;
    end
    
    imdb.train = 1:length(images_data);% 1:500;%:length(all_annos);
    imdb.val = imdb.train(1:3:end);
    train = setdiff(imdb.train,imdb.val);
    val = imdb.val;
    imdb.test = max([imdb.train imdb.val])+1:length(images_data);
    %     imdb.image_paths = images;
    if (isfield(imdb,'image_paths'))
        imdb = rmfield(imdb,'image_paths');
    end
    imdb.images = 1:length(images_data);
    imdb.images_data = images_data;
    imdb.labels = labels;
    imdb.nClasses = 15;
    save(opts.imdbPath,'imdb');
end

opts.numFetchThreads = 1 ; % not used yet
% training options (SGD)
opts.train.batchSize = 20;
%opts.train.numSubBatches = 5;
opts.train.numSubBatches = 5;
opts.train.continue = true ;
opts.train.gpus = 2;%[1 2];
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
gpuDevice(2);
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
%%
nEpochs = 50;
finalModelPath = fullfile(expDir,['net-epoch-' num2str(nEpochs) '.mat']);

% measure network performances at different epochs.
a = personmap.keys';
test_params.labels =newClassNames; % [{'none',a{:}}];
test_params.labels_to_block = [];
test_params.prefix = 'perfs_frac_10';
test_params.set = 'val';
train = imdb.train;val = imdb.val;test = imdb.test;
[perfs,diags] = test_net_perf(opts.expDir,50,imdb,train,val(1:10:end),[],test_params);
