function [feats] = getGeometricFeatures(polys,mouth_poly,face_poly)
faceBox = pts2Box(face_poly);
[r c] = BoxSize(faceBox);
face_scale = (r+c)/2;
feats.face_scale = face_scale;
mouthBox = pts2Box(mouth_poly);
faceCenter = mean([faceBox(1:2);faceBox(3:4)]);
mouthCenter = mean([mouthBox(1:2);mouthBox(3:4)]);



% since we know the polys are just rotated rectangles, we'll find the
% centers and scales using this information.
poly_centers = cellfun2(@mean ,polys);
poly_centers = cat(1,poly_centers{:});

feats.poly_centers = poly_centers;


poly_scales = cellfun2(@(x) max(sum(diff(x(1:3,:)).^2,2).^.5), polys);
poly_scales = cat(1,poly_scales{:});
% p2 = cellfun2(@(x) x(1,:), polys);
% p2 = cat(1,p2{:});
% poly_scales = 2*sum( (p1-poly_centers).^2,2).^.5;
feats.poly_scales = poly_scales;
% for k = 1:length(polys)
% %    [ U, mu, vars ] = pca( polys{k}' );
%     poly_scales(k) = mean(vars.^.5)*2;
%     poly_centers(k,:) = mu;
% end

face_box_poly = box2Pts(faceBox);
polys_area = zeros(size(polys));
face_area = polyarea(face_box_poly(:,1),face_box_poly(:,2));
intersection_area = zeros(size(polys));

newpolys = struct;
for k = 1:length(polys)
    newpolys(k).x = polys{k}(:,1);
    newpolys(k).y = polys{k}(:,2);
end

face_poly_xy.x = face_box_poly(:,1);face_poly_xy.y = face_box_poly(:,2);
polys_area = cellfun(@(x) polyarea(x(:,1),x(:,2)),polys);
for k = 1:length(polys)
    xy_poly = polys{k};
    x_a = PolygonClip(newpolys(k),face_poly_xy,1);
%     x_u = PolygonClip(newpolys(k),face_poly_xy,3);
%     [x_a,y_a] = polybool('and',xy_poly(:,1),xy_poly(:,2),face_box_poly(:,1),face_box_poly(:,2));
%     [x_u,y_u] = polybool('or',xy_poly(:,1),xy_poly(:,2),face_box_poly(:,1),face_box_poly(:,2));
%     polys_area(k) = polyarea(xy_poly(:,1),xy_poly(:,2));
    if (isempty(x_a))
        continue;
    end
    intersection_area(k) = polyarea(x_a.x,x_a.y);
    
    
end


% find the "top" of each poly.
% we know this to be the first edge, by construction in
% box2pts (a bit of a hack but saves me some other hassle)
polys_tops = cellfun2(@(x) x(1:2,:),polys);
tops_centers = cellfun2(@mean ,polys_tops);
tops_centers = cat(1,tops_centers{:});
poly_center_to_top = tops_centers-poly_centers; % vectors from center to top
feats.poly_center_to_top = poly_center_to_top;
feats.poly_center_to_top_n = normalize_vec(poly_center_to_top,2);
poly_center_to_face = bsxfun(@minus,faceCenter,poly_centers); % from poly center to face center
feats.poly_center_to_face = poly_center_to_face;
poly_top_to_face = bsxfun(@minus,faceCenter,tops_centers); % from poly center to face center
feats.poly_top_to_face = poly_top_to_face;
poly_top_to_mouth = bsxfun(@minus,mouthCenter,tops_centers); % from poly center to face center
feats.poly_top_to_mouth = poly_top_to_mouth;
feats.poly_center_to_face_n = normalize_vec(poly_center_to_face,2);
poly_center_to_mouth = bsxfun(@minus,mouthCenter,poly_centers);
feats.poly_center_to_mouth = poly_center_to_mouth;
feats.poly_center_to_mouth_n = normalize_vec(poly_center_to_mouth,2);
feats.poly_s_to_face_s = poly_scales/face_scale;
feats.intersection_of_polys = intersection_area./polys_area;
feats.intersection_of_faces = intersection_area./face_area;
feats.ovp = intersection_area./(polys_area+face_area-intersection_area);



end