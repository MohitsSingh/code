function bestRegion =  approximateRegion(target,regions,n,fast)

M = regionAdjacencyGraph(regions);
if (nargin < 3)
    n =  3;
end
if (nargin < 4)
    fast = 1;
end
groups = enumerateGroups(M,n);
% generate the regions...
bestOVP = 0;
bestGroup = 1;

if (~fast)
    
    for k = 1:length(groups)
        curGroup = groups{k};
        ii = k;
        for kk = 1:size(curGroup,1)
            curRegion = cat(3,regions{curGroup(kk,:)});
            curRegion = max(curRegion,[],3);
            intersection_ = nnz(target & curRegion);
            union_ = nnz(target | curRegion);
            if (union_ > 0)
                ovp = intersection_/union_;
            else
                ovp = 0;
            end
            if (ovp > bestOVP)
                bestOVP = ovp;
                bestGroup = curGroup(kk,:);
            end
        end
    end
    bestRegion = cat(3,regions{bestGroup});
    bestRegion=  max(bestRegion,[],3);
else
    newRegions = {};
    for k = 1:length(groups)
        curGroup = groups{k};
        ii = k;
        for kk = 1:size(curGroup,1)
            curRegion = cat(3,regions{curGroup(kk,:)});
            curRegion = max(curRegion,[],3);
            newRegions{end+1} = curRegion;
        end
    end
    ovp = boxRegionOverlap(target,newRegions);
    [o,io] = max(ovp);
    bestRegion = newRegions{io};
end