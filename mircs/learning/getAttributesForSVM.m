function [M,gt_labels_train] =  getAttributesForSVM(detsPrimary,detSecondary,gt_labels)
f = detsPrimary.cluster_locs(:,11);
gt_labels_train = gt_labels(f);
[~,~,~,~,M_primary] = calc_aps(detsPrimary,gt_labels);
M_primary = M_primary(f);
[~,~,~,~,M_secondary] = calc_aps(detSecondary,gt_labels(f));
M = [M_primary,M_secondary];
end