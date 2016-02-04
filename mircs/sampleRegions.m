function [regions,labels,ovps] = sampleRegions(regions,gt_regions,params)
%[regions,labels,ovps] = sampleRegions(regions,gt_regions,params)
% Samples some positive (overlapping with ground truth) and negative
% (non-overlapping with ground truth) regions given the parameters in
% params.


gt_regions(cellfun(@(x) isempty(x) || none(x(:)),gt_regions)) = [];
if isempty(gt_regions)
    sel_pos = [];
    if ~params.testMode
        sel_neg = vl_colsubset(1:length(regions),params.learning.nNegsPerPos,'uniform');    
    else
        sel_neg = 1:length(regions);
    end
    ovps = zeros(length(sel_neg));
else
    if ~iscell(gt_regions)
        gt_regions = {gt_regions};
    end
    
    if ~params.testMode && params.learning.include_gt_region
        regions = [regions;gt_regions];
        is_gt_region = false(size(regions));
        is_gt_region(end-length(gt_regions)+1:end) = true;
    else
        is_gt_region = false(size(regions));
    end
    [ovps,ints,uns] = regionsOverlap3(regions,gt_regions);    
    if strcmp(params.learning.ovpType,'intersection')
        ovps = ints./cellfun(@nnz,regions);
    elseif ~strcmp(params.learning.ovpType,'overlap')
        error('unexpected overlap type for ground truth assessment, expected ''intersection'' or ''overlap''');
    end
    
    if ~params.testMode    
        sel_neg = find(ovps < params.learning.negOvp);
        max_neg_to_keep = params.learning.max_neg_to_keep;
        sel_neg = vl_colsubset(sel_neg,params.learning.nNegsPerPos,'uniform');
        sel_pos = find(ovps >= params.learning.posOvp | is_gt_region);
        if (length(sel_neg) > max_neg_to_keep)
            sel_neg = vl_colsubset(row(sel_neg),max_neg_to_keep,'random')';
        end
    else % keep all regions in test mode, except the ground truth regions.
        sel_pos = find(ovps >= params.learning.posOvp & ~is_gt_region);
        sel_neg = find(ovps < params.learning.posOvp & ~is_gt_region);
    end
end
sel_ = [sel_pos;sel_neg];
regions = regions(sel_);
labels = ones(size(sel_));
labels(length(sel_pos)+1:end) = -1;
ovps = ovps(sel_);
end

