function [groups,newRegions] = expandRegions(regions,n,groups,M)
if (n==1)
    newRegions = regions;
    groups = {col(1:length(regions))};
end
if (nargin >= 2)    
    if (nargin < 3) || isempty(groups) % need to calculate...
        if (nargin < 4 || isempty(M))
            M = regionAdjacencyGraph(regions);
        end
        groups = enumerateGroups(M,n);
    end
end
% if (nargin ==4 && ~isempty(groups) && isempty(M))
%     M = regionAdjacencyGraph(regions);
% end
% if (nargin < 3 || isempty(groups))
%
%     M = regionAdjacencyGraph(regions);
%     groups = enumerateGroups(M,n);
% end
% if isempty(groups)
%     newRegions = regions;
%     return;
% end
if (nargout == 2)
    % generate the regions...
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
end