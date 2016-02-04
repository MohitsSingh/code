function gc_region_refined = refineRegionGC(im,rr,nIterations,toShow)
% rr = im2double(gc_regions{1});
if (nargin < 3)
    nIterations = 1;
end
if (nargin < 4)
    toShow = false;
end
if (none(rr))
    gc_region_refined = rr;
    return;
end
rr = im2double(rr);
bb = region2Box(rr);
bb = round(inflatebbox(bb,[2 2],'both',false));
gc_region_refined = cropper(rr,bb); I_sub = cropper(im,bb);
orig_sz = size(gc_region_refined);
% iterations....
for k = 1:nIterations
    curMask = imfilter(double(gc_region_refined),fspecial('gaussian',91,15));
    curMask = normalise(curMask);
    %curMask/max(curMask(:));
    
    gc_region_refined = getSegments_graphCut(I_sub,curMask,[],toShow);
    if (toShow) 
        drawnow; title(num2str(k)); pause;
    end
end

% displayRegions(I_sub,rr);
gc_region_refined = imresize(gc_region_refined,orig_sz);
gc_region_refined = shiftRegions(gc_region_refined,bb,im);
