function [clusters] = findNeighbours(conf,samples,locs,discovery_ids,suffix,toSave)
%     seedFeats = samples{seedIdx};


if (nargin < 6)
    toSave = 0;
end

if (nargin < 5)
    suffix = '';
    if (isfield(conf,'suffix'))
        suffix = conf.suffix;
    end
end

if (toSave)
    
    neighborsPath = fullfile(conf.cachedir,['neighbors' suffix '.mat']);
    if (exist(neighborsPath,'file'))
        load(neighborsPath);
        return;
    end
end

M = {};

n = length(discovery_ids);
n_samples = length(samples);
for k = 1:n_samples
    m = size(samples{k},2);
    M{k} = zeros(m,n);
    %         M{k}(:,k) = inf;
end

allLocs = cell(n_samples);
allFeats = cell(n_samples);

for k = 1:n
    imageID = discovery_ids{k};
    
    disp(['scanning images: %' num2str(100*k/n)]);
    
    I = toImage(conf,getImagePath(conf,imageID));
    [X,uu,vv,scales,t ] = allFeatures( conf,I );
    locs_ = uv2boxes(conf,uu,vv,scales,t);
    for seedIdx = 1:n_samples
%         seedIdx
        if (seedIdx == k)
            continue;
        end
        % find the nearest neighbor in image k for all features.
        seedFeats = samples{seedIdx};
        D = l2(seedFeats',X');
        [d,id] = min(D,[],2);
        allLocs{seedIdx,k} = locs_(id,:);
        allFeats{seedIdx,k} = X(:,id);
        M{seedIdx}(:,k) = d;
    end
end


re_locs = {};
re_samples = {};
for iImage = 1:n
    iImage
    M_ = M{iImage};
    cur_locs = allLocs(iImage,:);
    cur_samples = allFeats(iImage,:);
    cur_locs = cat(1,cur_locs{:});
    cur_samples = cat(2,cur_samples{:});
    m = size(M_,1);
    for iSample = 1:m % samples from this image.
        %         iSample
        %         iSample = IR(iSample_);
        % find the nearest neighbor for this patch.
        % location of nearest neighbor in other images
        sample_locs = cur_locs(iSample:m:end,:);
        % values of nearest neighbor in other images
        sample_X = cur_samples(:,iSample:m:end);
        % indices of other images.
        q = setdiff(1:size(M_,2),iImage);
        cur_dists = M_(iSample,q);
        [c,ic]  =sort(cur_dists,'ascend');
        % choose the top-5 neighbors for each sample.
        sample_locs(:,11) = q;
        sample_locs = sample_locs(ic(1:5),:);
        sample_locs = [locs{iImage}(iSample,:);sample_locs];
        sample_X = sample_X(:,ic(1:5));
        sample_X = [samples{iImage}(:,iSample),sample_X];
        re_locs{iImage,iSample} = sample_locs;
        re_samples{iImage,iSample} = sample_X;
        %         [p,p_mask] = visualizeLocs2(conf,discovery_ids,  re_locs{iImage,iSample} (1:5,:),64,1,0,0);
        %          imshow(multiImage(p));
        %          pause;
    end
end

re_locs = re_locs(:);
re_samples = re_samples(:);

clusters = struct('cluster_center',{},'cluster_samples',{},'cluster_locs',{});
clusterCount = 0;
for iCluster = 1:length(re_locs)
    current_re_locs = re_locs{iCluster};
    if (isempty(current_re_locs))
        continue;
    end
    current_re_samples = re_samples{iCluster};
    clusterCount = clusterCount + 1;
    clusters(clusterCount).cluster_center = mean(current_re_samples,2);
    clusters(clusterCount).cluster_samples = current_re_samples;
    clusters(clusterCount).cluster_locs = current_re_locs;
    % initialize the classifer as the mean of the
    % samples initialized to the current cluster
    clusters(clusterCount).w = clusters(clusterCount).cluster_center;
    clusters(clusterCount).b = 0;
    clusters(clusterCount).sv = [];
    nSamples = size(clusters(clusterCount).cluster_samples,2);
    %     if (nClusters == size(samples,2))
    %         clusters(clusterCount).isvalid = true;
    %     else
    clusters(clusterCount).isvalid = ...
        nSamples >= conf.clustering.min_cluster_size;
    %     end
    clusters(clusterCount).cluster_samples = ...
        clusters(clusterCount).cluster_samples(:,1:min(nSamples,conf.clustering.top_k));
    clusters(clusterCount).cluster_locs = ...
        clusters(clusterCount).cluster_locs(1:min(nSamples,conf.clustering.top_k),:);
end
if (toSave)
    save(neighborsPath,'clusters');
end

end