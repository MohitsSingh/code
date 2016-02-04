function [ features] = getImageFeatures( all_feats,req_ids )
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
all_ids = {all_feats.imageID};
[lia,lib] = ismember(req_ids,all_ids);
all_feats = all_feats(lib);
features = struct('feats',{},'extent',{},'layer',{});
%feat_extent = {'feats_full','feats_crop','feats_face','feats_face_ext'};
feat_extent = setdiff(fieldnames(all_feats),'imageID');
p = 0;
for iExtent = 1:length(feat_extent)
    curExtent = feat_extent{iExtent};
    for q = 1:size(all_feats,2)
        if (isempty(all_feats(q).(curExtent)))
            all_feats(q).(curExtent) = zeros(4096,1);
        end
    end    
    p = p+1;
    features(p).feats = cat(2,all_feats.(curExtent));
    features(p).extent = curExtent;
    %features(p).layer = feat_layers(iLayer);
end
% end
end