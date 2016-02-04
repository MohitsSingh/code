function [labels,features,ovps,is_gt_region] = collectFeaturesFromImg(conf,imgData,params)
%resPath = j2m(params.dataDir,imgData);
resPath = j2m(params.featsDir,imgData);
if (~exist(resPath,'file'))
    [regionFeats,imageFeats,selected_regions] = extract_all_features(conf,imgData,params)
    save(resPath,'regionFeats','imageFeats');
end

L = load(resPath);%,'moreData'); % don't need the segmentation here...
% if isfield(L
%     L = load(resPath);
%     feats = L.regionFeats;
%     moreData = L.imageFeats;
%     L = struct('feats',feats,'moreData',moreData);
    
% end
% matObj = matfile(resPath);
% a = whos(matObj);
% if (~strcmp(a(1).name,'feats'))
%     L = struct('feats',matObj.regionFeats,'moreData',matObj.imageFeats);
% else
%     L = matObj;
% end
% % load(resPath,'feats','moreData');

[labels,features,ovps,is_gt_region] = collectFeatures(L,params.features);