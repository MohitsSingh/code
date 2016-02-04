function clustering2(conf,discovery_sets,natural_set)
% initial clustering.
% discovery_ids =cat(1,discovery_sets{:});
% before the first iteration, sample from the first set
% windows at multiple scales.
% before the first iteration, use k-means to cluster patches
% from the discovery set.
% if (nargin > 3)
%     conf.dict = dict;
% end
ids_true_train = col(cat(1,discovery_sets{:}));
ids_false_train = vl_colsubset(col(natural_set)',length(ids_true_train),'Uniform');

%
distsFileName = [conf.suffix 'dists12.mat'];
if (~exist(distsFileName,'file'))
    dists12 = imageSetDistances_new(conf,ids_true_train,ids_false_train{1});
    save(distsFileName,'dists12');
else
    load(distsFileName);
end
%%
suffix = [];
if (isfield(conf,'suffix'))
    suffix = conf.suffix;
end

clustersPath = fullfile(conf.cachedir,['kmeans_clusters_vis' suffix '.mat']);
if (~exist(clustersPath,'file'))
    [samples,locs,~] = findDiscriminativePatches(conf,ids_true_train,ids_false_train,dists12);
    
    samples_cat = cat(2,samples{:});
    [neighbors,allLocs,allFeats] = findNeighbors2(conf,samples_cat,ids_true_train,[],0,1);
    conf.clustering.top_k = 5;
    clusters = neighbors2clusters(conf,neighbors,allLocs,allFeats,false);
    if (conf.level < 2)
        conf = get_clustering_conf(conf);
    end
    
    clusters = clusters([clusters.isvalid]);
    
    % clusters = refineClusters(conf,discovery_ids,clusters);
    
    if (~isempty(conf.debug.cluster_choice))
        valids = find([clusters.isvalid]);
        clusters = clusters(valids(1:min(length(valids),conf.debug.cluster_choice)));
    end
    suffix = '';
    if (isfield(conf,'suffix'))
        suffix = conf.suffix;
    end
    
    for k = 1:length(clusters) %TODO - this was inserted to retain only the 1-nn
        clusters(k).cluster_samples = clusters(k).cluster_samples(:,1);
        clusters(k).cluster_locs = clusters(k).cluster_locs(1,:);
    end
    
    save(clustersPath,'clusters');
else
    load(clustersPath);
end
%

% if (demo_mode) % visualize the initial clusters...
imgPath = fullfile(conf.demodir,['iter_0' suffix '.jpg'])
if (~exist(imgPath,'file'))

%     clustersPath = fullfile(conf.cachedir,['kmeans_clusters_vis' suffix '.mat']);
%     if (~exist(clustersPath,'file'))
        [clusters,allImgs] = visualizeClusters(conf,ids_true_train,clusters,64);
%         save(clustersPath,'clusters');
%     else
%         load(clustersPath);
%     end

    m = clusters2Images(clusters([clusters.isvalid]));
    imwrite(m(1:min(60000,end),:,:),imgPath);
end
% end

% clusters = clusters(1:5);% TODO!!
for iter = 1:conf.clustering.num_iter
    
    % train classifiers for current clusters.
    
    clusters = train_patch_classifier(conf,clusters,natural_set,iter);
    clusters = rmfield(clusters,'sv');
    val_set = get_alternate_set(conf,discovery_sets,iter);
    
    topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
    % % %
    if (exist(topDetectionsPath,'file'))
        load(topDetectionsPath);
    else
        
        conf.detection.params.detect_save_features = 1;
        mwe= conf.detection.params.detect_max_windows_per_exemplar;
        conf.detection.params.detect_max_windows_per_exemplar = 1;
        detections = getDetections(conf,val_set,clusters,iter);
        conf.detection.params.detect_max_windows_per_exemplar = mwe;
        conf.detection.params.detect_save_features = 0;
        % remove all but the top k (e.g, 5) detections for each
        % cluster.
        % now we are left with new cluster centers and A,
        % which contains the top detection window HOGS which
        % can serve as positive examples for the next iteration.
        
        [clusters] = getTopDetections(conf,detections,clusters);
%         clusters = rmfield(clusters,'sv');
        
        save(topDetectionsPath,'clusters','-v7.3');
        
    end
    
    %     if (demo_mode)
    [clusters,allImgs] = visualizeClusters(conf,val_set,clusters,64);
    m = clusters2Images(clusters);
    imwrite(m(1:min(60000,end),:,:),fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']));
    clusters = rmfield(clusters,'vis');
    %     end
end

%
% valids = []; % for visualization purposes, retain only the clusters
% % which "survived" all iterations
% for iter = 1:conf.clustering.num_iter
%     topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
%     load(topDetectionsPath);
%     valids = [valids;[clusters.isvalid]];
% end
%
% isvalid = sum(valids,1) == size(valids,1);
%
% disp(sprintf('Total of %d / %d valid clusters after %d iterations\n',...
%     sum(isvalid),length(isvalid),conf.clustering.num_iter));
%
% for iter = 1:conf.clustering.num_iter
%     topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
%     load(topDetectionsPath);
%     c = visualizeClusters(conf,get_alternate_set(conf,discovery_sets,iter),clusters(isvalid),64);
%     m = clusters2Images(c);
%
%     imwrite(m(1:min(60000,end),:,:),fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']));
% end
%
% I = [];
% for iter = 1:conf.clustering.num_iter
%     I = cat(2,I,imread(fullfile(conf.demodir,sprintf('iter_%02.0f%s.jpg',iter,suffix))));
% end
% imwrite(m(1:min(60000,end),:,:),fullfile(conf.demodir,['iters' suffix '.jpg']));
