function [ features] = getImageFeatures_2(feat_matrix)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
features = struct('feats',{},'extent',{});
%feat_extent = {'feats_full','feats_crop','feats_face','feats_face_ext'};
p = 0;
for iExtent = 1:size(feat_matrix,2)
    features(iExtent).extent = feat_matrix(1,iExtent).type;
    features(iExtent).feats = cat(2,feat_matrix(:,iExtent).feat);
end
% end
end