function [regions,toKeep] = removeDuplicateRegions(regions,removeEmpty)

if (nargin < 2)
    removeEmpty = true;
end
if (removeEmpty)
    regions = regions(cellfun(@nnz,regions)>0);
end

if (isempty(regions))
    return;
end

% first sort by number of elements.
areas = cellfun(@nnz,regions);
a = unique(areas);

toKeep = {};
for k = 1:length(a)
    f = find(areas==a(k));
    if (length(f)==1)
        toKeep{end+1} = f;
    else
        r = cellfun2(@row,regions(f));
        r = cat(1,r{:});
        [r,m] = unique(r,'rows');
        %         if (length(f) ~= length(m))
        %             'aha'
        %             k
        %         end
        toKeep{end+1} = col(f(m));
    end
end
toKeep = cat(1,toKeep{:});
regions = regions(toKeep);