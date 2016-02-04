
%function clustering(conf,ids,labels)

function clustering(conf,discovery_sets,natural_sets,dict)
% initial clustering.
[discovery_set,natural_set] =get_alternate_set(conf,discovery_sets,natural_sets,0);
% before the first iteration, sample from the first set
% windows at multiple scales.
% before the first iteration, use k-means to cluster patches
% from the discovery set.
if (nargin > 3)
    conf.dict = dict;
end
[samples,locs] = sampleHogs(conf,discovery_set);
% initial clustering using k-means
demo_mode = conf.demo_mode;
samples = cat(2,samples{:});
locs = cat(1,locs{:});
[clusters] = kMeansClustering(conf,samples,locs,1);
% TODO - note that the minimal scale is .4 and that
% there is no pyramid padding. This is to avoid "black" areas
% and all the related artifacts.
if (conf.level < 2)
    conf = get_clustering_conf(conf);
end

clusters = clusters([clusters.isvalid]);


if (~isempty(conf.debug.cluster_choice))
    valids = find([clusters.isvalid]);
    clusters = clusters(valids(1:min(length(valids),conf.debug.cluster_choice)));
end
suffix = '';
if (isfield(conf,'suffix'))
    suffix = conf.suffix;
end
if (demo_mode) % visualize the initial clusters...
    imgPath = fullfile(conf.demodir,['iter_0' suffix '.jpg'])
    if (~exist(imgPath,'file'))
        
        clustersPath = fullfile(conf.cachedir,['kmeans_clusters_vis' suffix '.mat']);
        if (~exist(clustersPath,'file'))
            [clusters_,allImgs] = visualizeClusters(conf,discovery_set,clusters);
            save(clustersPath,'clusters');
        else
            load(clustersPath);
        end
        
        m = clusters2Images(clusters_([clusters.isvalid]));
        imwrite(m,imgPath);
    end
end

% clusters = clusters(1:5);% TODO!!
for iter = 1:conf.clustering.num_iter
    
    % train classifiers for current clusters.
    
    clusters = train_patch_classifier(conf,clusters,natural_set,iter);
    % save memory by clearing sv's
    clusters = rmfield(clusters,'sv');
    
    % fire the detector on the held out set.
    [discovery_set,natural_set] = get_alternate_set(conf,discovery_sets,natural_sets,iter);
    
    topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
    % % %
    if (exist(topDetectionsPath,'file'))
        load(topDetectionsPath);
    else
        
        conf.detection.params.detect_save_features = 1;
        mwe= conf.detection.params.detect_max_windows_per_exemplar;
        conf.detection.params.detect_max_windows_per_exemplar = 1;
        detections = getDetections(conf,discovery_set,clusters,iter);
        conf.detection.params.detect_max_windows_per_exemplar = mwe;
        conf.detection.params.detect_save_features = 0;
        % remove all but the top k (e.g, 5) detections for each
        % cluster.
        % now we are left with new cluster centers and A,
        % which contains the top detection window HOGS which
        % can serve as positive examples for the next iteration.
        
        [clusters] = getTopDetections(conf,detections,clusters);
        
        save(topDetectionsPath,'clusters','-v7.3');
    end
    
    %     if (demo_mode)
    [clusters,allImgs] = visualizeClusters(conf,discovery_set,clusters);
    m = clusters2Images(clusters);
    imwrite(m,fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']));
    %     end
end
% % % 
% % % 
% % % valids = []; % for visualization purposes, retain only the clusters
% % % % which "survived" all iterations
% % % for iter = 1:conf.clustering.num_iter
% % %     topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
% % %     load(topDetectionsPath);
% % %     valids = [valids;[clusters.isvalid]];
% % % end
% % % 
% % % isvalid = sum(valids,1) == size(valids,1);
% % % 
% % % disp(sprintf('Total of %d / %d valid clusters after %d iterations\n',...
% % %     sum(isvalid),length(isvalid),conf.clustering.num_iter));
% % % 
% % % for iter = 1:conf.clustering.num_iter
% % %     topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
% % %     load(topDetectionsPath);
% % %     c = visualizeClusters(conf,get_alternate_set(conf,discovery_sets,iter),clusters(isvalid));
% % %     m = clusters2Images(c);
% % %     
% % %     imwrite(m,fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']));
% % % end
% % % 
% % % I = [];
% % % for iter = 1:conf.clustering.num_iter
% % %     I = cat(2,I,imread(fullfile(conf.demodir,sprintf('iter_%02.0f%s.jpg',iter,suffix))));
% % % end
% % % imwrite(I,fullfile(conf.demodir,['iters' suffix '.jpg']));
