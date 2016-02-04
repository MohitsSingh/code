function res = extract_straw_feats(mouth_corners, poly_centers, angles)
res = zeros(length(mouth_corners),4);

mouth_centers = cellfun2(@mean,mouth_corners);
mouth_centers = cat(1,mouth_centers{:});

% res = bsxfun(@rdivide,poly_centers-mouth_centers,faceScales);
res = [poly_centers-mouth_centers sind(angles) cosd(angles)];

% for t = 1:length(mouth_corners)
%     tl = boxes(t,1:2);
%     curCorners = mouth_corners{t};
%     mouth_center = mean(curCorners);
% %     left_corner = curCorners(1,:);
% %     u = curCorners(2,:)-curCorners(1,:);
% %     u = u/norm(u);
% %     v = [-u(2),u(1)];
%     %poly_center = mean(poly_centers{t});
%     poly_center = poly_centers(t,:);
% 
%     %p = (poly_center-left_corner)/faceScales(t);
%     p = (poly_center-mouth_center)/faceScales(t);
%     res(t,:) = [p sind(angles(t)) cosd(angles(t))];
% end
end