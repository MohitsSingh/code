function clusters = getClusterSeeds(conf,ids,toSave,varargin)
% sample semi-dense patches from positive set
% clusters = initClusters;

if (toSave)
    clustersPath = fullfile(conf.cachedir,['initial_clusters_' conf.suffix '.mat']);
    if (exist(clustersPath,'file'))
        load(clustersPath);
        return;
    end
end

ip = inputParser;
ip.addParamValue('minScale',.5,@isnumeric);
ip.addParamValue('maxScale',1,@isnumeric);
ip.addParamValue('ovp',1/3,@isnumeric);
ip.parse(varargin{:});


conf.detection.params.detect_min_scale  = ip.Results.minScale;
conf.detection.params.detect_max_scale = ip.Results.maxScale;
conf.detection.params.detect_levels_per_octave = 3;
[posSamples,posLocs] = samplePatches(conf,ids,ip.Results.ovp);

% [clusters] = kMeansClustering(conf,posSamples,posLocs,false,[]);

% [a,aa] = visualizeClusters(conf,ids,clusters);
% imwrite(clusters2Images(a),'clust1.jpg');

clusters = makeClusters(posSamples,posLocs);

% do_flip = 1;
% addLocation = 0;%TODO  - add this as a configuration stage...
% keepFeats = true;
% getQE =false;
% conf.detection.params.detect_min_scale = .5;
% conf.detection.params.detect_max_scale = 1;
% conf.detection.params.detect_levels_per_octave = 5;
% save_samples = 0;
% suffix = [];
% 
% 

% [M,allLocs,allFeats] = findNeighbors2(conf,posSamples,ids,suffix...
%     ,save_samples,do_flip,getQE,addLocation,keepFeats)

% top_k = 1;
% clusters = neighbors2clusters(conf,M,allLocs,allFeats,false,top_k);
% 
if (toSave)
    save(clustersPath,'clusters');
end
end
 
