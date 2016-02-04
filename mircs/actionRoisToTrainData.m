function action_rois_poly = actionRoisToTrainData(action_rois,true_ids)
    action_rois_poly = struct('sourceImage',{},'polygon',{});
    for k = 1:length(true_ids)
        action_rois_poly(k).sourceImage = true_ids{k};
        xmin = action_rois(k,1); ymin = action_rois(k,2);
        xmax = action_rois(k,3); ymax = action_rois(k,4);
        action_rois_poly(k).polygon.x = [xmin xmax xmax xmin];
        action_rois_poly(k).polygon.y = [ymin ymin ymax ymax];
        action_rois_poly(k).partID = 1;
    end
end