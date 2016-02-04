function clusters=refineClusters_circulant(conf,clusters,discovery_sets,natural_sets,suffix,varargin)
[~,natural_set] = get_alternate_set(conf,discovery_sets,natural_sets,0);

% conf.detection.params.max_models_before_block_method = 10;
for iter = 1:conf.clustering.num_iter
    
    % train classifiers for current clusters using the circulant
    % decomposition
    job_suffix = sprintf('detect_%d_%s',iter,suffix);
    
    topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
    % % %
    if (exist(topDetectionsPath,'file') && ~conf.debug.override)
        load(topDetectionsPath);
    else
        
        clusters_path = fullfile(conf.cachedir,[job_suffix '.mat']);
        if (exist(clusters_path,'file') && ~conf.debug.override)
            load(clusters_path);
            disp('loaded clusters')
            %clusters = train_circulant(conf,clusters,natural_set);                       
        else
            if conf.parallel
                clusters = launch_train_circulant_parallel(conf,clusters,natural_set,job_suffix);
            else
                clusters = train_circulant(conf,clusters,natural_set);
            end
            disp('done training new clusters');
            save(clusters_path,'clusters');
        end
        if (isempty(clusters))
            delete(clusters_path);
            error(['clusters empty: ' clusters_path]);
        end
        w = conf.features.winsize;
        %% figure,imagesc(hogDraw(reshape(clusters(1).w,w(1),w(2),[]),15,1));
        
        % fire the detector on the held out set.
        [discovery_set,natural_set] = get_alternate_set(conf,discovery_sets,natural_sets,iter);
        
        
        conf.detection.params.detect_save_features = 1;
        mwe= conf.detection.params.detect_max_windows_per_exemplar;
        conf.detection.params.detect_max_windows_per_exemplar = 1;
        dkt = conf.detection.params.detect_keep_threshold;
        conf.detection.params.detect_keep_threshold = 0;
        %         tic
        
        
        if (conf.parallel)
            detections = launchDetectParallel2(conf,clusters,discovery_set,job_suffix);
        else            
            detections = getDetections(conf,discovery_set,clusters,iter,suffix,false);
            detections = detections';
        end
                
        % showDetections(discovery_set,detections)
        conf.detection.params.detect_keep_threshold = dkt;
        conf.detection.params.detect_max_windows_per_exemplar = mwe;
        conf.detection.params.detect_save_features = 0;
        
        % remove all but the top k (e.g, 5) detections for each
        % cluster.
        % now we are left with new cluster centers and A,
        % which contains the top detection window HOGS which
        % can serve as positive examples for the next iteration.
        
        [clusters] = getTopDetections(conf,detections,clusters,'uniqueImages',true);
        clusters = removeInvalidClusters(clusters);
        save(topDetectionsPath,'clusters','-v7.3');
        demo_mode = 1;
        if (demo_mode)
            demo_file = fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']);
            %         if (~exist(demo_file,'file'))
            [clusters,allImgs] = visualizeClusters(conf,discovery_set,clusters,'height',...
                64,'disp_model',true,'add_border',false,'nDetsPerCluster',50);
            m = clusters2Images(clusters);
            %         figure; imagesc(m); axis image;
            imwrite(m(1:min(60000,size(m,1)),:,:),demo_file);
        end
    end
    
    conf.clustering.top_k = conf.clustering.top_k+1;
end
end