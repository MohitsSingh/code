
function [clusters,conf] = train_patch_classifier(conf,clusters,neg_images,varargin)
% suffix,C,w1,toSave)

% train a detector for each of the clusters defined by samples indexed by
% A, using the cluster-assigned samples as positive examples and patches
% samples from the images in natural_set as negative examples.

% TODO - make sure than no patches in the negative examples set
% are identical to those in the positives...

ip = inputParser;
ip.addParamValue('suffix','',@(x) ischar(x));
ip.addParamValue('C',.01,@isnumeric);%.01
ip.addParamValue('w1',1,@isnumeric);% w1 50
ip.addParamValue('toSave',true,@islogical);
ip.addParamValue('keepSV',true,@islogical);
ip.addParamValue('override',false,@islogical);
ip.addOptional('iter',[],@isnumeric);

ip.parse(varargin{:});

C = ip.Results.C;
w1 = ip.Results.w1;
toSave = ip.Results.toSave;
keepSV = ip.Results.keepSV;
suffix = ip.Results.suffix;
iter = ip.Results.iter;
override =  ip.Results.override;

detectors_path = fullfile(conf.cachedir,['detectors_' num2str(iter) suffix '.mat']);

if (isfield(clusters,'vis'))
    clusters = rmfield(clusters,'vis');
end

if (toSave && exist(detectors_path,'file') && ~override)
    load(detectors_path);
else
    det_params = conf.detection.params;
    det_params.detect_save_features = 1;
    det_params.detect_max_windows_per_exemplar = 50; 
    det_params.detect_exemplar_nms_os_threshold = 1.0;
    % memory limit per iteration
    max_variable_size_bytes = .1*10^9; % (100 MB)
    valids  = find([clusters.isvalid]); % process only valid clusters
    bytes_per_window = 8*size(clusters(1).cluster_samples(:,1),1); % double precision w
    nImages_per_mine = 5;
    bytes_per_mine = nImages_per_mine*det_params.detect_max_windows_per_exemplar * ...
        bytes_per_window;
    % the X2 factor is due to image flipping
    nClusters_per_chunk = 2*floor(max_variable_size_bytes/bytes_per_mine);        
    nClusters_per_chunk = 50;
    
    % for each cluster, the other clusters serve as negative samples.
    %
    %     for k = 1:length(valids)
    %         clusters(valids(k)).sv = [];
    %     end
    
    if (conf.clustering.split_clusters)
        cluster_samples = {};
        for k = 1:length(valids)
            cluster_samples{k} = clusters(valids(k)).cluster_samples;
        end
    end
    
    models = {};
    
    % get some hard-negatives for the clusters...
    
    tt = randperm(length(neg_images));
    for i = 1:conf.clustering.num_hard_mining_iters
        d = nImages_per_mine;
        for k = 1:length(valids)
            model = [];
            winsize = conf.features.winsize;
            cluster_id = valids(k);
            % if this is the first iteration, train the classifier using the
            % other clusters...
            if (i == 1)
                
                if (conf.clustering.split_clusters)
                    %error('split clusters not supported...');
                    %neg_samples = cluster_samples(:,inds_~=k);
                    neg_samples = cat(2,cluster_samples{[1:k-1 k+1:end]});
                    pos_samples = clusters(cluster_id).cluster_samples;
                    [svm_model,ws,b,sv,coeff] = train_classifier(pos_samples,neg_samples,C,w1);
                    clusters(cluster_id).sv = full(sv);
                else
                    %ws = clusters(cluster_id).w;%mean(clusters(cluster_id).cluster_samples,2);
                  ws = clusters(cluster_id).cluster_samples(:,1);
                    b = clusters(cluster_id).b;
                end
                model.w = ws;
                model.b =  b;
            else
                model.w = clusters(cluster_id).w;
                model.b = clusters(cluster_id).b;
            end
            
            model.w = reshape(model.w,winsize(1),winsize(2),[]);
            model.hg_size = size(model.w);
            model.init_params = det_params.init_params;
            models{k}.models_name = 'clustering';
            models{k}.model = model;
        end
        
        
        % break the detection into chunks so that each chunk doesn't
        % occupy too much memory.
        
        chunks = 1:nClusters_per_chunk:length(valids);
        cur_imageInds = (i-1)*d+1:min(length(tt),i*d);
        cur_images = tt(cur_imageInds);
        for iChunk = 1:length(chunks)
            chunkStart = chunks(iChunk);
            if (iChunk == length(chunks))
                chunkEnd = length(valids);
            else
                chunkEnd = chunks(iChunk+1)-1;
            end
            cur_models = models(chunkStart:chunkEnd);
            
            xs = {};
%             profile on;
            for iImage = 1:length(cur_images)
                currentID = neg_images{cur_images(iImage)};
                %                 disp(currentID);
                if (ischar(currentID))
                    I = getImage(conf,currentID);
                else
                    I = currentID;
                end
                [rs,~] = esvm_detect(im2double(I),cur_models,det_params);
                
                % keep at most 50 hard negatives from each image.
                for ix = 1:length(rs.xs)
                    rs.xs{ix} = rs.xs{ix}(:,1:min(50,size(rs.xs{ix},2)));
                end
                xs{iImage} = rs.xs;
            end
%             profile viewer;
            xs = cat(2,xs{:});
            if (isempty(xs))
                continue; %TODO - this should be a continue, no?
            end
            
            for iCluster = chunkStart:chunkEnd
                disp(['clusters done: ' num2str(100*iCluster/length(valids)) '%%']);
                cluster_id = valids(iCluster);
                neg_samples = cat(2,xs{iCluster-chunkStart+1,:});
                if (isempty(neg_samples))
                    continue;
                end
                if (iscell(neg_samples))
                    neg_samples =cat(2,neg_samples{:});
                end
                pos_samples = clusters(cluster_id).cluster_samples;
                if (isfield(clusters(cluster_id),'sv')) % adding the negative
                    %                 support vectors as negatives...
                    if (keepSV)
                        neg_samples = [neg_samples,clusters(cluster_id).sv];
                    end
                end
                disp(['neg samples: ' num2str(size(neg_samples,2))]);
                clear sv; % so we don't accumulate from previous iterations...!
                
%                 classifier = train_classifier_pegasos(pos_samples,neg_samples,1);
                [svm_model,w,b,sv,coeff] = train_classifier(pos_samples,neg_samples,C,w1);
                % default parameters before...
%                 w = classifier.w(1:end-1);
%                 b = classifier.w(end);
                clusters(cluster_id).w = w;
                clusters(cluster_id).b =  b;
                if (keepSV)
                    if (~isempty(coeff))
                        sv = sv(:,coeff<0);
                        clusters(cluster_id).sv = full(sv);
                    end                    
                end
                models{iCluster}.model.w = reshape(w,winsize(1),winsize(2),[]);
                models{iCluster}.model.b = b;
                
            end
            
            if (cur_imageInds(end) == length(tt))
                break;
            end
        end
    end
    if (toSave)
        if (isfield(clusters,'vis'))
            clusters = rmfield(clusters,'vis');
        end
        if (~keepSV && isfield(clusters,'sv'))
            clusters = rmfield(clusters,'sv');
        end
        save(detectors_path,'clusters','conf','-v7.3');
    end
end
end
