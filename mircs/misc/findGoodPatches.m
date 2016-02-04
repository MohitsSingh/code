function [clusters,estimatedQuality] = findGoodPatches(conf,pos_set,neg_set,toSave)
%FINDGOODPATCHES Summary of this function goes here
%   Detailed explanation goes here

if (toSave)
    clustersPath = fullfile(conf.cachedir,['initial_clusters' conf.suffix '.mat']);
    if (exist(clustersPath,'file'))
        load(clustersPath);
        return;
    end
end


[pos_sets,neg_sets] = split_ids(conf,ids,labels);

pos_train = pos_sets{1};
neg_train = neg_sets{1};

clusters = getClusterSeeds(conf,pos_train);

% use knn to estimate which are the good patches.

pos_val = pos_sets{2};
neg_val = neg_sets{2};
clusterQuality = estimateDiscriminativePower(conf,clusters,pos_val,neg_val);

if (toSave)
    save(clustersPath,'clusters','estimatedQuality');
end


% [c,ic] = sort(clusterQuality,'ascend');
% clusters_ = visualizeClusters(conf,pos_train,clusters(ic(1:10)),...
%     64,0,1)

% imwrite(clusters2Images(clusters_),'new1.jpg');