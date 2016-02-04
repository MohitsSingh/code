function [gc_segResult,obj_box] = checkSegmentation(I,obj_poly,hard_poly)
% examine a low-level segmentation versus the given object mask to see
% if the expected boundaries of the object are found in the image.

if numel(obj_poly) == prod(size2(I))
    obj_mask = obj_poly;
    obj_box = region2Box(obj_mask);    
else
    obj_box = pts2Box(obj_poly);
    obj_box = round(inflatebbox(obj_box,2));    
    obj_mask = poly2mask2(obj_poly,size2(I));
end
I_sub = cropper(I,obj_box);
obj_mask = cropper(obj_mask,obj_box);
if (nargin < 3)
    obj_mask_hard = [];
else
    obj_mask_hard = poly2mask2(hard_poly,size2(I));
    obj_mask_hard = cropper(obj_mask_hard,obj_box);
end
gc_segResult = getSegments_graphCut_2(I_sub,obj_mask,[],0,obj_mask_hard);
%
%clf,imagesc();axis image;
%     drawnow;pause

