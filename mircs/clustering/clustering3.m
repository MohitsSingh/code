
%function clustering(conf,ids,labels)
function clustering3(conf,discovery_sets,natural_sets,varargin)
% initial clustering.
[discovery_set,natural_set] =get_alternate_set(conf,discovery_sets,natural_sets,0);
discovery_set = discovery_set(1:10:end);
%[clusters,estQuality] = findGoodPatches(conf,discovery_set,natural_set,1);
ip = inputParser;
ip.addParamValue('clusteringConf',get_clustering_conf(conf),@isstruct);
ip.addParamValue('ovp',1/3,@isnumeric);
ip.parse(varargin{:});
clusters = getClusterSeeds(conf,discovery_set,1,'minScale',1,'ovp',ip.Results.ovp);

%[samples,locs] = sampleHogs(conf,discovery_set);
% initial clustering using k-means
demo_mode = conf.demo_mode;

% if (conf.level < 2)
%     conf = get_clustering_conf(conf);
% end
conf = ip.Results.clusteringConf;

clusters = clusters([clusters.isvalid]);

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

refineClusters(conf,clusters,discovery_sets,natural_sets,conf.suffix,'keepSV',false);

