function [bbox_feats,B,regionBoxes] = get_bbox_feats(regions,bbox,imgSize,kp_locs)
regionBoxes = cellfun2(@region2Box,regions);
regionBoxes = cat(1,regionBoxes{:});
[bbox_feats.heights bbox_feats.widths bbox_feats.areas] = BoxSize(regionBoxes);
[ bbox_feats.overlaps ,ints] = boxesOverlap(regionBoxes,bbox);
%     ints = BoxIntersection2(regionBoxes,bbox);
[h,w,a] = BoxSize(bbox);
bbox_feats.ints_in_a = ints./a;
bbox_feats.ints_in_b = ints./bbox_feats.areas;
bbox_feats.rel_areas = bbox_feats.areas/a;

bbox_feats.hasLeft = regionBoxes(:,1) < bbox(1);
bbox_feats.hasTop = regionBoxes(:,2) < bbox(2);
bbox_feats.hasRight = regionBoxes(:,3) > bbox(3);
bbox_feats.hasBottom = regionBoxes(:,4) > bbox(4);
bbox_feats.norm_coords = bsxfun(@rdivide,regionBoxes,[imgSize imgSize]);

kps_inbox = zeros(size(regionBoxes,1),size(kp_locs,1));
rel_coords_x = zeros(size(kps_inbox));
rel_coords_y = zeros(size(kps_inbox));
regionBoxCenters = boxCenters(regionBoxes);
for u = 1:size(kp_locs)
        x = kp_locs(u,1);y = kp_locs(u,2);
        kps_inbox(:,u) = x >= regionBoxes(:,1) & y >=regionBoxes(:,2) & x <= regionBoxes(:,3) & y <= regionBoxes(:,4);    
        rel_coords_x(:,u) = regionBoxCenters(:,1)-x;
        rel_coords_y(:,u) = regionBoxCenters(:,2)-y;
end


bbox_feats.kps_inbox = kps_inbox;
bbox_feats.centerDistances = l2(regionBoxCenters,kp_locs).^.5/imgSize(1);
bbox_feats.rel_coords_x = rel_coords_x/imgSize(1);
bbox_feats.rel_coords_y = rel_coords_y/imgSize(2);


B = [bbox_feats.heights/h bbox_feats.widths/w bbox_feats.rel_areas bbox_feats.ints_in_a...
    bbox_feats.ints_in_b bbox_feats.overlaps bbox_feats.hasLeft bbox_feats.hasTop bbox_feats.hasRight...
    bbox_feats.hasBottom bbox_feats.norm_coords kps_inbox bbox_feats.centerDistances ...
    bbox_feats.rel_coords_x bbox_feats.rel_coords_y];
end