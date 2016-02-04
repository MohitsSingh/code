function [overlaps,inds] = analyzeOverlaps(all_img_inds,all_boxes,gt_boxes,images)
n = size(gt_boxes,1);
overlaps = zeros(1,n);
inds = zeros(1,n);
for t = 1:length(all_img_inds)
    f = find(all_img_inds==t);
    if (isempty(f))
        continue
    end
    curBoxes = all_boxes(f,:);
    ovps = boxesOverlap(curBoxes,gt_boxes(t,:));
    [r,ir] = max(ovps);
    overlaps(t) = r;
    inds(t) = f(ir);    
end
end
