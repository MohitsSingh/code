function [feats,labels] = loadStageResults(conf,all_results,fra_db,stage2Dir,featureParams);

clear feats;
% labels = {};
for k = 1:5%length(all_results)
    k
    if (isempty(all_results(k).labels))
        continue
    end
    %     break
    orig_img = fra_db(k);
    resPath = j2m(stage2Dir,orig_img);
    L = load(resPath,'feats','moreData');
    feats(k) = L;
    %     labels = L.feats;
end
[all_labels, all_features,all_ovps,is_gt_region,orig_inds] = collectFeatures(featStruct,featureParams)