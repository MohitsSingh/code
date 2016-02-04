function [ious_u,ious] = bestCoveringScore(VOCopts,ids,res)
ious_u = zeros(size(ids));
ious = zeros(size(ids));
for k = 1:length(ids)
    k
    fg_mask_ = imread(sprintf(VOCopts.seg.clsimgpath,ids{k}));
    dc_mask = fg_mask_ ==255;
    gt_mask = fg_mask_ > 0 & ~dc_mask;
    r = res{k} > 0;
    %   r = r &~ dc_mask;
    
    
    % calculate the intersection over union with the ground-truth; create
    % multiple
    if (nargout == 2)
        rprops_my = regionprops(r,'PixelIdxList','Area');
        rprops_gt = regionprops(gt_mask,'PixelIdxList','Area');
        nPixels = 0;
        curScore = 0;
        for ii = 1:length(rprops_gt)
            currentRegion = rprops_gt(ii).PixelIdxList;
            nPixels = nPixels + length(currentRegion);
            cur_ious= zeros(1,length(rprops_my));
            for jj = 1:length(rprops_my)
                int_ = intersect(rprops_my(jj).PixelIdxList,...
                    currentRegion);
                u_ = union(rprops_my(jj).PixelIdxList,...
                    currentRegion);
                cur_ious(jj) = length(int_)/length(u_);
            end
            curScore = curScore+length(currentRegion)*max(cur_ious);
        end
        if (isempty(curScore))
            curScore = 0;
        end
        ious(k) = curScore/nPixels;
    end
    r = r(:);
    gt_mask = gt_mask(:);
    intersection_ = sum(r & gt_mask);
    union_ = sum(r | gt_mask);
    ious_u(k) = intersection_ / union_;
    
end