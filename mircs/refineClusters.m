function clusters=refineClusters(conf,clusters,discovery_sets,natural_sets,suffix,varargin)
[~,natural_set] = get_alternate_set(conf,discovery_sets,natural_sets,0);

ip = inputParser;
ip.addParamValue('keepSV',false,@islogical);

ip.parse(varargin{:});
keepSV = ip.Results.keepSV;
conf.detection.params.max_models_before_block_method = 10;
for iter = 1:conf.clustering.num_iter
    
    % train classifiers for current clusters.
%      clusters.cluster_samples = clusters.cluster_samples(:,1);
    clusters = train_patch_classifier(conf,clusters,natural_set,iter,'toSave',true,...
        'C',.01,'w1',50,'suffix',suffix,'keepSV',keepSV,'override',true);
    
    w = conf.features.winsize;
    %% figure,imagesc(hogDraw(reshape(clusters(1).w,w(1),w(2),[]),15,1));
    % save memory by clearing sv's??
    
    if (isfield(clusters,'sv') && ~keepSV)
        clusters = rmfield(clusters,'sv');
    end
    if (isfield(clusters,'vis'))
        clusters = rmfield(clusters,'vis');
    end
    
    % fire the detector on the held out set.
    [discovery_set,natural_set] = get_alternate_set(conf,discovery_sets,natural_sets,iter);    
    topDetectionsPath = fullfile(conf.cachedir,['top_detections_' num2str(iter) suffix '.mat']);
    % % %
    if (exist(topDetectionsPath,'file') && 0)
        load(topDetectionsPath);
    else
        
        conf.detection.params.detect_save_features = 1;
        mwe= conf.detection.params.detect_max_windows_per_exemplar;
        conf.detection.params.detect_max_windows_per_exemplar = 1;
        detections = getDetections(conf,discovery_set,clusters,iter,suffix,false);                                                
        % showDetections(discovery_set,detections)
        
        conf.detection.params.detect_max_windows_per_exemplar = mwe;
        conf.detection.params.detect_save_features = 0;
        % remove all but the top k (e.g, 5) detections for each
        % cluster.
        % now we are left with new cluster centers and A,
        % which contains the top detection window HOGS which
        % can serve as positive examples for the next iteration.
        
        [clusters] = getTopDetections(conf,detections,clusters,'uniqueImages',true);
        
        
        save(topDetectionsPath,'clusters','-v7.3');
    end
    
    conf.clustering.top_k = conf.clustering.top_k+1;
    
    demo_mode = 1;
    if (demo_mode)
        demo_file = fullfile(conf.demodir,['iter_' sprintf('%02.0f',iter) suffix '.jpg']);
%         if (~exist(demo_file,'file'))
            [clusters,allImgs] = visualizeClusters(conf,discovery_set,clusters,'height',...
                64,'disp_model',true,'add_border',false,'nDetsPerCluster',10);
            m = clusters2Images(clusters);
            figure; imagesc(m); axis image;
            imwrite(m(1:min(60000,size(m,1)),:,:),demo_file);
%         end
    end
end