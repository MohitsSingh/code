% % 'beer bottle'
% %     'beer can'
% %     'beer glass'
% %     'beer mug'
% %     'bottle'
% %     'coffee cup'
% %     'coffee mug'
% %     'cup'
% %     'Dixie cup'
% %     'glass'
% %     'water glass'
% %     'mug'
% %     'wine bottle'
%% Experiment 0027
%% Feb 5, 2014

%% learn parts of objects which could specify interaction. Search for such parts in a small area around the face region.
%% This is similar to experiment 26 but this time, the negative examples are from the face area of the non-action
%% class.

if (~exist('initialized','var'))
    initpath;
    config
    init_hog_detectors;
    
    conf.demodir = '~/mircs/clustering/demo';
    false_images_path = fullfile(conf.cachedir,'false_for_disc_patches.mat');
    if (exist(false_images_path,'file'))
        load(false_images_path);
    else
        % false images set - all images of non-drinking class...
        [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
        %         load ~/storage/misc/imageData_new;
        naturalSets = {};
        false_images = {};
        imageSet = imageData.train;
        minFaceScore = -.6;
        conf.get_full_image = true;
        for k = 1:length(imageSet.imageIDs)
            k
            if (~validImage(imageSet,k,false,minFaceScore))
                continue;
            end
            if (imageSet.labels(k)),continue;end
            currentID = imageSet.imageIDs{k};
            m = getSubImage(conf,newImageData,currentID);
            m = imresize(m,[240 NaN],'bilinear');
            bb = [1 1 size(m,2) size(m,2)];
            dd = .7;
            bb = round(clip_to_image(inflatebbox(bb,[dd dd],'both',false),m));
            false_images{end+1} = cropper(m,bb);
        end
        save(false_images_path,'false_images');
    end
    
    conf.get_full_image = false;
    conf.get_full_image = true;
    naturalSets{1} = false_images(1:2:end);
    naturalSets{2} = false_images(2:2:end);
    
    % conf.features.winsize = conf.features.winsize + 2;
    
    %     conf.features.winsize = [5 5];
    
    initialized = true;
end
% end

conf.features.winsize = [6 6];
conf.features.padSize = 3;
conf.detection.params.init_params.sbin = 8;
conf.features.winsize = conf.features.winsize + conf.features.padSize;
add_suffix = sprintf('_%d_%d_top_sbin_%d',conf.features.winsize,conf.detection.params.init_params.sbin);
nElementsPerCluster = 30;
conf.useCirculant = true;
conf.parallel = false;
conf.debug.override = false;
wsz = conf.features.winsize;
for iClass = 1:length(classifiers)
    
    currentClass = classifiers(iClass).class;
    % was the initial sampling saved for this class?
    cachePath = fullfile(conf.cachedir,['samples_' currentClass add_suffix '.mat']);
    trainedPath =  fullfile(conf.cachedir,[currentClass add_suffix '_trained.mat']);
    %     images = dataset_list(dataset,'train',currentClass);
    %     goods = false(size(images));
    %     for k = 1:length(images)
    %         %         k
    %         images{k} = sprintf(dataset.VOCopts.imgpath,images{k});
    %         goods(k) = exist(images{k},'file');
    %     end
    %     images = images(goods);
    %
    if (~exist(trainedPath,'file'))
        if (~exist(cachePath,'file'))
            [initialSamples,locs,all_ims,images] = collect_object_samples_2(conf,dataset,currentClass);
            norms = sum(initialSamples.^2,1).^.5;
            %             showSorted(all_ims,-norms);
            initialSamples = initialSamples(:,norms > 3);
            all_ims = all_ims(norms > 3);
            save(cachePath,'initialSamples','all_ims','images');
        else
            load(cachePath);
        end
        nSamples = size(initialSamples,2);
%         [IC,C] = kmeans2(initialSamples',1,...
%             struct('nTrial',10,'outFrac',.1,...
%             'display',1,'minCl',3));
        
        
        [IC,C] = kmeans2(initialSamples',round(nSamples/nElementsPerCluster),...
            struct('nTrial',10,'outFrac',.1,...
            'display',1,'minCl',3));
        outliers = IC == -1;
        fprintf('fraction of outliers: %0.3f\n',nnz(outliers)/length(IC));
        %     IC(outliers) = [];
        %     initialSamples(:,outliers) = [];
        %     all_ims(outliers) =[];
        maxPerCluster = .7;
        [curClusters,ims]= makeClusterImages(all_ims',C',IC',initialSamples,[],maxPerCluster);
        %         [curClusters]= makeClusterImages([],C',IC',initialSamples,[],maxPerCluster);
        
        %         discoverySets(1).images = images(1:2:end);
        %         discoverySets(2).images = images(2:2:end);
        discoverySets{1} = images(1:2:end);
        discoverySets{2} = images(2:2:end);
        suffix = [ 'disc_patches_'  add_suffix classifiers(iClass).name];
        conf.clustering.top_k = 10;
        conf.detection.params.max_models_before_block_method = 0;
        conf.detection.params.detect_add_flip = 1;
        conf.detection.params.detect_min_scale = .1;
        conf.max_image_size = inf;
        if (conf.useCirculant)
            clusters = refineClusters_circulant(conf,curClusters,discoverySets,naturalSets,suffix);
        else
            clusters = refineClusters(conf,curClusters,discoverySets,naturalSets,suffix);
        end
        close all
        save(trainedPath,'clusters');
    end
end

