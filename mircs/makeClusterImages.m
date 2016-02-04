function [clusters,ims,inds] = makeClusterImages(imgs,C,IC,X,outDir,maxPerCluster)
if (~isempty(outDir))
    ensuredir(outDir);
end
uic = unique(IC(IC~=-1)); % remove outliers
if (nargin < 6)
    maxPerCluster = inf;
end
clusters = makeClusters(C,[]);
ims = {};
imss ={};
inds = {};
for k = 1:length(uic);
    k    
    clusterElements = find(IC==uic(k));
    %clusterElements = clusterElements(1:min(length(clusterElements),maxPerCluster));
    if (~isempty(X))
        curX = X(:,clusterElements);    
        distToCenter = l2(C(:,k)',curX');
        [~,iDist] = sort(distToCenter,'ascend');
    else
        curX = zeros(1,length(clusterElements));
        iDist = 1:length(curX);
    end
    mm = maxPerCluster;
    if (mm < 1)
        mm = round(length(iDist)*mm);
    end
    iDist = iDist(1:min(length(iDist),mm));
    clusters(k).cluster_samples = curX(:,iDist);
    inds{k} = clusterElements(iDist);
    if (isempty(iDist))
        error(['empty!!' num2str(k)]);
    end
    if (~isempty(imgs))
        a = imgs(inds{k});
        ims{k} = multiImage(a,false);
        if (~isempty(outDir))
            imwrite(ims{k},fullfile(outDir,sprintf('%03.0f.jpg',k)));
        end
    end
end
end