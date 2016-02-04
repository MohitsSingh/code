initpath;
config;
conf.features.winsize = 8;
conf.class_subset = DRINKING;
conf.max_image_size = 256;
conf.detection.params.detect_min_scale = .5;
conf.detection.params.detect_levels_per_octave = 10;
%
conf.clustering.min_cluster_size = 1;
conf.clustering.split_discovery = false;


for iClass = 9:length(A)
    % dataset of images with ground-truth annotations
    % start with a one-class case.
    conf.class_subset = iClass;
    currentSuffix=  [A{iClass} '_all'];
    [train_ids,train_labels] = getImageSet(conf,'train',1,0);
    discovery_set = train_ids(train_labels);
    natural_set = train_ids(~train_labels);
    ids_true_train = col(cat(1,discovery_set));
    
%     ids_true_train = ids_true_rtrain;
    ids_false_train = vl_colsubset(col(natural_set)',length(ids_true_train),'Uniform');
    
    
    samplesPath = fullfile('/home/amirro/storage/data/dists/',[currentSuffix '_samples_train.mat']);
    
    if (~exist(samplesPath,'file'))
        
        distsPath = fullfile('/home/amirro/storage/data/dists/',[currentSuffix '_dists_train.mat']);
        
        if (~exist(distsPath,'file'))
            dists12 = imageSetDistances(conf,ids_true_train,ids_false_train);
            save(distsPath,'dists12');
        else
            load(distsPath);
        end
        
        [samples,locs,~] = findDiscriminativePatches(conf,ids_true_train,ids_false_train,dists12,1);
        save(samplesPath,'samples','locs');
    else
        load(samplesPath);
    end
   visPath = fullfile('/home/amirro/storage/data/dists/',[currentSuffix '_vis_train.jpg']);
   
   [p] = visualizeLocs2(conf,ids_true_train,cat(1,locs{:}),64,1,0,0);
   imwrite(multiImage(p),visPath);
    
end