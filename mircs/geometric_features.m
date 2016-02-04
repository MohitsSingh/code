function G = geometric_features(ref_rect,target_rects)
[rows,cols,a1] = BoxSize(ref_rect);
[~,~,a2] = BoxSize(target_rects);
boxes_i = BoxIntersection(ref_rect,target_rects);
[~,~,ints] = BoxSize(boxes_i);
ints = ints(:);
%[~, ~, bi] = BoxSize(boxes_i(has_intersection,:));
obj_in_face = ints./a2;
face_in_obj = ints/a1;
ovps = ints./(a1+a2-ints);
bc = boxCenters(target_rects);
bc_face = boxCenters(ref_rect);
bc_diff = bsxfun(@minus,bc,bc_face);
bc_diff = bc_diff/rows;
G = [obj_in_face,face_in_obj,ovps,bc_diff];
end