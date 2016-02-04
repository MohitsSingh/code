function P = estimateDiscriminativePower(conf,clusters,pos_val,neg_val)

clusterSamples = cat(2,clusters.cluster_samples);

cluster_inds = {};
for k = 1:length(clusters)
    nElements = size(clusters(k).cluster_samples,2);
    cluster_inds{k} = k*ones(1,nElements);
end

cluster_inds = cat(2,cluster_inds{:});

keepFeats = false;
suffix = [];
save_samples = false;
do_flip = 1;
qetQE = 0;
addLocation = 0;
[M_pos] = findNeighbors2(conf,clusterSamples,pos_val,suffix...
    ,save_samples,do_flip,qetQE,addLocation,keepFeats);

neg_val = neg_val(vl_colsubset(1:length(neg_val),3*length(pos_val),'Uniform'));

[M_neg] = findNeighbors2(conf,clusterSamples,neg_val,suffix...
    ,save_samples,do_flip,qetQE,addLocation,keepFeats);

M_pos_min = min(M_pos,[],2);
M_neg_min = min(M_neg,[],2);

P = zeros(1,length(clusters));

for k = 1:length(P)
    q = find(cluster_inds==k);
    P(k) = mean(M_pos_min(q)./(M_pos_min(q)+M_neg_min(q)));
end