function predictionErrors = measurePredictionError(preds_xy,all_kps)

nKeypoints = size(preds_xy,2);
predictionErrors = {};

for iKeypoint = 1:nKeypoints
    cur_gt = squeeze(all_kps(:,iKeypoint,:));
%     goods = find(~any(isnan(cur_gt),2));
    %r = squeeze(preds_xy(goods,iKeypoint,:));
    r = squeeze(preds_xy(:,iKeypoint,:));
    predictionErrors{iKeypoint} = sum((cur_gt-r).^2,2);
    
%     errors(iKeypoint) = E;
end
