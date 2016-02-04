function [all_features,all_labels,all_ovps,all_img_inds] = collect_feature_subset(conf,imgs,params,max_neg_to_keep)

% featureData = struct('img_ind',{},'label',{},'ovp',{},'feats',{});

all_features = {};
all_labels = {};
all_ovps = {};
all_img_inds = {};
if (nargin <  3)
    max_neg_to_keep = inf;
end

for iImg = 1:length(imgs)
    iImg
    % load features
    if (~imgs(iImg).valid)
        continue
    end
    [labels,features,ovps,is_gt_region] = collectFeaturesFromImg(conf,imgs(iImg),params);
    % select a feature subset
    sel_neg = find(ovps < params.learning.negOvp);
    sel_neg = vl_colsubset(sel_neg,params.learning.nNegsPerPos,'uniform');
    if (length(sel_neg) > max_neg_to_keep)
        sel_neg = vl_colsubset(sel_neg,max_neg_to_keep,'random');
    end
    sel_pos = find(ovps >= params.learning.posOvp | is_gt_region);
    pos_labels = ones(1,length(sel_pos))*imgs(iImg).classID;
    all_labels{end+1} = [-ones(1,length(sel_neg)) pos_labels];
    all_features{end+1} = features(:,[sel_neg sel_pos]);
    all_ovps{end+1} = ovps([sel_neg sel_pos]);
    all_img_inds{end+1} = iImg*ones(size(all_ovps{end}));
end
% all_labels = cat(2,all_labels{:});
% all_features = cat(2,all_features{:});
% all_ovps = cat(2,all_ovps{:});