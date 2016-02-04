function [mouth_img,mouth_rect,mouth_pts,xy_near_mouth] = get_mouth_img(I_orig,xy_global,curPose,I,resizeFactor,imgSize)
if (abs(curPose) > 30)
    error('face angles larger than 30 not handled for now')
end
mouth_corner_inds = [35 41];
mouth_pts = xy_global(mouth_corner_inds,:);
mouth_center = mean(mouth_pts);
mouth_rect = inflatebbox([mouth_center mouth_center],[1.5 1]*size(I,1)/(resizeFactor*3),'both',true);
mouth_rect = inflatebbox(mouth_rect,1.5,'both',false);
poly_source = box2Pts(mouth_rect);
[mouth_img,T] = rectifyWindow(I_orig,round(poly_source),imgSize);
xy_near_mouth = tformfwd(T,xy_global);
end