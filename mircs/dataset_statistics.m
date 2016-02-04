% produce some statistics about the fra_db dataset

% load fra_db;
goods = true(size(fra_db));
obj_sizes = zeros(size(fra_db));
img_sizes = zeros(size(fra_db));
for t = 1:length(fra_db)
    t
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),defaultROIParams);
    iObj = find(strncmpi('obj',{rois.name},3));
    gt_masks = {};
    bads = false(size(iObj));
    if ~isempty(iObj)
        for ii = 1:length(iObj)
            curPoly = rois(iObj(ii)).poly;
            Z = poly2mask2(curPoly,size2(I));
            obj_sizes(t) = max(obj_sizes(t),nnz(Z));
%             displayRegions(I,Z);
        end
    else
        goods(t) = false;
    end
end

% hist(obj_sizes(goods & obj_sizes>0),logspace(1,5))
%%
for t = 54:length(fra_db)
%     if (~goods(t) || obj_sizes(t) == 0),continue,end
%     if (obj_sizes(t) > 200),continue,end
%     if (obj_sizes(t) < 30),continue,end
    t
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),defaultROIParams);
    iObj = find(strncmpi('obj',{rois.name},3));
    gt_masks = {};
%     bads = false(size(iObj));
    if ~isempty(iObj)
        for ii = 1:length(iObj)
            curPoly = rois(iObj(ii)).poly;
            Z = poly2mask2(curPoly,size2(I));
%             obj_sizes(t) = max(obj_sizes(t),nnz(Z));
            displayRegions(I,Z);title(num2str(obj_sizes(t)));
        end
    else
%         goods(t) = false;
    end
    pause
end




regionSizes = {};
nRegions = {};
for t = 1:10:length(fra_db)
    t
    [rois,roiBox,I,scaleFactor,roiParams] = get_rois_fra(conf,fra_db(t),defaultROIParams);
    ppp = j2m('~/storage/s40_fra_feature_pipeline_partial_dnn_deep',fra_db(t));
    if ~exist(ppp,'file'),continue,end
    L = load(ppp,'selected_regions');
    regionSizes{end+1} = cellfun(@nnz,L.selected_regions);
end
nRegions = cellfun(@numel,regionSizes);
regionSizes1 = cat(1,regionSizes{:});

    iObj = find(strncmpi('obj',{rois.name},3));
    gt_masks = {};
%     bads = false(size(iObj));
    if ~isempty(iObj)
        for ii = 1:length(iObj)
            curPoly = rois(iObj(ii)).poly;
            Z = poly2mask2(curPoly,size2(I));
%             obj_sizes(t) = max(obj_sizes(t),nnz(Z));
            displayRegions(I,Z);title(num2str(obj_sizes(t)));
        end
    else
%         goods(t) = false;
    end
    pause
end
