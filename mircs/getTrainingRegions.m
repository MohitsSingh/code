function all_data = getTrainingRegions(all_data,selectionParams)
min_pos_gt = selectionParams.min_pos_gt;
max_neg_gt = selectionParams.max_neg_gt;
maxNegPerImage = selectionParams.maxNegPerImage;
for t = 1:length(all_data)
    regions = all_data(t).regions;
    curLabels = zeros(1,length(regions));
    curLabels(all_data(t).gt_ovp > min_pos_gt) = 1;
    curLabels(all_data(t).gt_ovp < max_neg_gt) = -1;
    sel_pos = find(curLabels==1);
    sel_neg = find(curLabels==-1);
    sel_neg = vl_colsubset(sel_neg,maxNegPerImage);
    sel_ = [sel_pos,sel_neg];
    all_data(t).regions = regions(sel_);
    all_data(t).labels = curLabels(sel_);    
    all_data(t).gt_ovp = all_data(t).gt_ovp(sel_);
end
end