function [occludingRegions,occlusionPatterns,rprops] = getOccludingCandidates(im,occlusionPattern)

occludingRegions = occlusionPattern.regions;
% face_mask = poly2mask2(curImageData.face_poly,size2(im));
face_mask = occlusionPattern.face_mask;

% map regions using face coordinate system.
% figure,imshow(im); hold on; plotPolygons(curImageData.face_poly);

% check distance of each point on perimeter to each segment.
% face_poly_r = round(curImageData.face_poly);
% dist_signature = zeros(size(curImageData.face_poly,1),length(occludingRegions));
% for k = 1:length(occludingRegions)
%     bw = bwdist(occludingRegions{k});
%     dist_signature(:,k) = bw(sub2ind2(size(bw),fliplr(face_poly_r)));
% end

occlusionPatterns = occlusionPattern.occlusionPatterns;
[f1,f2] = BoxSize(occlusionPattern.faceBox);
face_scale = (f1+f2)/2;
mouth_dist_n = [occlusionPatterns.min_dist_to_mouth]/face_scale;
% displayRegions(im,occludingRegions,-mouth_dist_n);
region_sel = mouth_dist_n <= .15;
occludingRegions = occludingRegions(region_sel);
occlusionPatterns = occlusionPatterns(region_sel);
RR = directionalROI(im,mean(occlusionPattern.mouth_poly,1),[0 1]',140);
RR = RR | face_mask;
[ovp,ints,uns] = regionsOverlap(occludingRegions,RR);
areas = cellfun(@nnz,occludingRegions(:));
region_sel = ints./areas >=.8;
occludingRegions = occludingRegions(region_sel);
occlusionPatterns = occlusionPatterns(region_sel);
% figure,imagesc(im); axis image; hold on; plotPolygons(curImageData.face_poly);plotPolygons(curImageData.mouth_poly);
% displayRegions(im,RR);
% displayRegions(im,occludingRegions);

% displayRegions(im,occludingRegions,[occlusionPatterns.face_in_seg],0);
% filter occluders according to some rules.
region_sel = [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < .5 &...
    [occlusionPatterns.area_rel_face] <= 5;
occludingRegions = occludingRegions(region_sel);
occlusionPatterns = occlusionPatterns(region_sel);
if (none(region_sel))
    rprops = [];
    disp('no regions left for this image');
    return;
end
%[rprops,x] = extractShapeFeatures(fillRegionGaps(occludingRegions));
[rprops,x] = extractShapeFeatures(occludingRegions);
% first two rows of x are centroids (x,y)
% last 2 rows of x are cos(theta) sin(theta) of orienation.
region_sel = [rprops.MajorAxisLength]./[rprops.MinorAxisLength] >= 1.5;
occludingRegions = occludingRegions(region_sel);
occlusionPatterns = occlusionPatterns(region_sel);
rprops = rprops(region_sel);
x = x(:,region_sel);
% occludingRegions = occludingRegions(c1); rprops = rprops(c1); x = x(:,c1);
% displayRegions(im,occludingRegions,[rprops.Orientation]);
x_ = x(1,:);y_=x(2,:);
u = x(end-1,:); v = -x(end,:);
% quiver(x_,y_,u,v);
%
% imagesc(im); hold on; plotBoxes(curImageData.lipBox);
% find the vector from the center of the object to the
mouth_center = mean(occlusionPattern.mouth_poly,1);
vecToMouth = bsxfun(@plus,-[x_(:) y_(:)],mouth_center);
[vecToMouth] = normalize_vec(vecToMouth,2);
% figure,imagesc(im); hold on;axis image;plotPolygons(curImageData.mouth_poly);
% quiver(x_(region_sel),y_(region_sel),u(region_sel),v(region_sel),'r');
% quiver(x_(region_sel),y_(region_sel),vecToMouth(region_sel,1)',vecToMouth(region_sel,2)','g');
dots = abs(sum(vecToMouth.*[u(:) v(:)],2));
for k = 1:length(dots)
    rprops(k).dot = dots(k);
end
% displayRegions(im,occludingRegions,dots,0,5);