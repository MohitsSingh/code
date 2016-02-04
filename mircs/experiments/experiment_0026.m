%% Experiment 0026
%% Jan 29, 2014

%% learn parts of objects which could specify interaction. Search for such parts in a small area around the face region.

add_suffix = '_2';
if (~exist('initialized','var'))
    initpath;
    config
    init_hog_detectors;
    conf.demodir = '~/mircs/clustering/demo_new';
%     mkdir(conf.demodir);
    % false images set - all images of non-drinking class...
    [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
    conf.detection.params.detect_min_scale = .1;
    naturalSets = {};
    false_images = train_ids(~train_labels);
    false_images = false_images(1:10:end);
    conf.get_full_image = false;
    conf.max_image_size = 256;
    
    for k = 1:length(false_images)
        k
        false_images{k} = getImage(conf,false_images{k});
        %     false_images{k} = fullfile(conf.imgDir,false_images{k});
        
    end
    conf.get_full_image = true;
    naturalSets{1} = false_images(1:2:end);
    naturalSets{2} = false_images(2:2:end);
    
    % conf.features.winsize = conf.features.winsize + 2;
    conf.features.winsize = [5 5];
    wsz =conf.features.winsize;
    initialized = true;
end
for iClass = 1:length(classifiers)
    currentClass = classifiers(iClass).class;
    % was the initial sampling saved for this class?
    cachePath = fullfile(conf.cachedir,['samples_' currentClass '.mat']);
    trainedPath =  fullfile(conf.cachedir,[currentClass '_trained.mat']);
    
    images = dataset_list(dataset,'train',currentClass);
    goods = false(size(images));
    for k = 1:length(images)
        %         k
        images{k} = sprintf(dataset.VOCopts.imgpath,images{k});
        goods(k) = exist(images{k},'file');
    end
    images = images(goods);
    
    if (~exist(trainedPath,'file'))
        if (~exist(cachePath,'file'))
            [initialSamples,all_ims] = collect_object_samples(conf,dataset,currentClass);
            
          
            [~,sel_] = removeInvalidClusters(curClusters,5);
            
            norms = sum(initialSamples.^2,1).^.5;
            %             showSorted(all_ims,norms);
            initialSamples = initialSamples(:,norms > 3);
            all_ims = all_ims(norms > 3);
            
               [IDX,M] = meanShift(initialSamples', 2);
             size(initialSamples)
             size(M)
            [curClusters,ims]= makeClusterImages(all_ims,M',IDX,initialSamples,[],maxPerCluster);
            
            save(cachePath,'initialSamples','all_ims');
        else
            load(cachePath);
        end
        nSamples = size(initialSamples,2);
        [IC,C] = kmeans2(initialSamples',round(nSamples/10),struct('nTrial',1,'outFrac',.1,...
            'display',1,'minCl',3));
        outliers = IC == -1;
        fprintf('fraction of outliers: %0.3f\n',nnz(outliers)/length(IC));
        %     IC(outliers) = [];
        %     initialSamples(:,outliers) = [];
        %     all_ims(outliers) =[];
        maxPerCluster = .7;
        %         [curClusters,ims]= makeClusterImages(all_ims',C',IC',initialSamples,[],maxPerCluster);
        [curClusters]= makeClusterImages([],C',IC',initialSamples,[],maxPerCluster);
        
        
        discoverySets{1} = images(1:2:end);
        discoverySets{2} = images(2:2:end);
        suffix = [ 'disc_patches_'  classifiers(iClass).name];
        %     c_.cluster_samples= c_.cluster_samples(:,1);
        conf.clustering.top_k = 5;
        conf.detection.params.max_models_before_block_method = 0;
        conf.detection.params.detect_add_flip = 0;
        conf.detection.params.detect_min_scale = .1;
        conf.max_image_size = 256;
        clusters = refineClusters_circulant(conf,curClusters,discoverySets,naturalSets,suffix);
        close all
        save(trainedPath,'clusters');
    end    
end


% add_suffix = '2';
% % if (~exist('initialized','var'))
% %     initpath;
% %     config
% %     init_hog_detectors;
% %     
% %     conf.demodir = '~/mircs/clustering/demo';
% %     false_images_path = fullfile(conf.cachedir,'false_for_disc_patches.mat');
% %     if (exist(false_images_path,'file'))
% %         load(false_images_-path);
% %     else
% %         % false images set - all images of non-drinking class...
% %         [train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
% %         %         load ~/storage/misc/imageData_new;
% %         naturalSets = {};
% %         M = {};
% %         imageSet = imageData.train;
% %         minFaceScore = -.6;
% %         conf.get_full_image = true;
% %         for k = 1:length(imageSet.imageIDs)
% %             k
% %             if (~validImage(imageSet,k,false,minFaceScore))
% %                 continue;
% %             end
% %             if (imageSet.labels(k)),continue;end
% %             
% %             %     break
% %             currentID = imageSet.imageIDs{k};
% %             curIndex = findImageIndex(newImageData,currentID);
% %             [I,I_rect] = getImage(conf,currentID);
% %             [xy_c,mouth_poly,face_poly] = getShiftedLandmarks(newImageData(curIndex).faceLandmarks,-I_rect);
% %             face_box = pts2Box(face_poly);
% %             face_box = inflatebbox(face_box,2.5,'both',false);
% %             face_poly = bsxfun(@minus,face_poly,face_box(1:2));
% %             mouth_poly = bsxfun(@minus,mouth_poly,face_box(1:2));
% %             %             newImageData(curIndex).sub_image =
% %             M{end+1} = im2uint8(min(1,max(0,cropper(I,round(face_box)))));
% %         end
% %         
% %         false_images = M;
% %         save(false_images_path,'false_images');
% %     end
% %     
% %     conf.get_full_image = false;
% %     conf.max_image_size = 256;
% %     conf.get_full_image = true;
% %     naturalSets{1} = false_images(1:2:end);
% %     naturalSets{2} = false_images(2:2:end);
% %     
% %     % conf.features.winsize = conf.features.winsize + 2;
% %     conf.features.winsize = [5 5];
% %     wsz =conf.features.winsize;
% %     initialized = true;
% % end
% % % end
% % for iClass = 1:length(classifiers)
% %     currentClass = classifiers(iClass).class;
% %     % was the initial sampling saved for this class?
% %     cachePath = fullfile(conf.cachedir,['samples_' currentClass '.mat']);
% %     trainedPath =  fullfile(conf.cachedir,[currentClass add_suffix '_trained.mat']);
% %     
% %     images = dataset_list(dataset,'train',currentClass);
% %     goods = false(size(images));
% %     for k = 1:length(images)
% %         %         k
% %         images{k} = sprintf(dataset.VOCopts.imgpath,images{k});
% %         goods(k) = exist(images{k},'file');
% %     end
% %     images = images(goods);
% %     
% %     if (~exist(trainedPath,'file'))
% %         if (~exist(cachePath,'file'))
% %             [initialSamples,all_ims] = collect_object_samples(conf,dataset,currentClass);
% %             norms = sum(initialSamples.^2,1).^.5;
% %             %             showSorted(all_ims,norms);
% %             initialSamples = initialSamples(:,norms > 3);
% %             all_ims = all_ims(norms > 3);
% %             save(cachePath,'initialSamples','all_ims');
% %         else
% %             load(cachePath);
% %         end
% %         nSamples = size(initialSamples,2);
% %         [IC,C] = kmeans2(initialSamples',round(nSamples/10),struct('nTrial',1,'outFrac',.1,...
% %             'display',1,'minCl',3));
% %         outliers = IC == -1;
% %         fprintf('fraction of outliers: %0.3f\n',nnz(outliers)/length(IC));
% %         %     IC(outliers) = [];
% %         %     initialSamples(:,outliers) = [];
% %         %     all_ims(outliers) =[];
% %         maxPerCluster = .7;
% %         %         [curClusters,ims]= makeClusterImages(all_ims',C',IC',initialSamples,[],maxPerCluster);
% %         [curClusters]= makeClusterImages([],C',IC',initialSamples,[],maxPerCluster);
% %         
% %         
% %         discoverySets{1} = images(1:2:end);
% %         discoverySets{2} = images(2:2:end);
% %         suffix = [ 'disc_patches_'  add_suffix classifiers(iClass).name];
% %         %     c_.cluster_samples= c_.cluster_samples(:,1);
% %         conf.clustering.top_k = 5;
% %         conf.detection.params.max_models_before_block_method = 0;
% %         conf.detection.params.detect_add_flip = 1;
% %         conf.detection.params.detect_min_scale = .1;
% %         conf.max_image_size = 256;
% %         clusters = refineClusters_circulant(conf,curClusters,discoverySets,naturalSets,suffix);
% %         close all
% %         save(trainedPath,'clusters');
% %     end
% % end

