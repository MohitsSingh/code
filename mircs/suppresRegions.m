function [subsets] = suppresRegions(regionOvp,T_ovp,scores,I,regions)
% Do region-based non-maxima suppression. For a set of regions whose
% overlaps are defined by regionOvp, pick greedily the top scoring regions
% according to <scores>. scores is a kxn matrix, where each row refers to a
% different score. Hence several subsets may be created.

nRegions = size(regionOvp,1);
if (nargin < 3)
    scores = ones(1,nRegions,1);
end

if size(scores,2)==1
    scores = row(scores);
end

[scores,iScores] = sort(scores,2,'descend');
subsets = {};
for r = 1:size(scores,1)
    ovp = regionOvp(iScores(r,:),iScores(r,:));
    curSel = suppressHelper(ovp);
    subsets{r} = iScores(r,curSel');
end

if length(subsets)==1
    subsets = subsets{1};
end

    function sel_ = suppressHelper(ovp)
        sel_ = false(size(ovp,1),1);
        for zz = 1:length(sel_)
            curOvp = ovp(zz,sel_);
            if (any(curOvp))
                if (max(curOvp) >= T_ovp)
                    continue;
                end
            end
            sel_(zz) = true;
        end
    end
end
