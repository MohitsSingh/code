function [occlusionPatterns,regions,face_mask,mouth_mask] = ...
    getOcclusionPattern(conf,imgData,varargin)
% calculate geometric relations between expected face mask and other
% regions to indicate if they occlude the face.

ip = inputParser;
ip.addParamValue('toExpand',0);
ip.addParamValue('G',[]);
ip.addParamValue('regions',[]);
ip.addParamValue('roi',[]);
ip.addParamValue('useGCRegions',false);
ip.parse(varargin{:});
G = ip.Results.G;
regions = ip.Results.regions;
toExpand = ip.Results.toExpand;
useGCRegions = ip.Results.useGCRegions;
roi = ip.Results.roi;
if (imgData.faceScore==-1000)
    occlusionPatterns = [];
    angularCoverage = [];
    face_mask = [];
    mouth_mask = [];
    region_sel = [];
    regions = [];
    return;
end
currentID = imgData.imageID;
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
ref_box = [0 0];
if (isempty(roi))
    roi = [1 1 dsize(I,[2 1])];
end
if (conf.get_full_image)
    ref_box = [xmin ymin xmax ymax];
end
[xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imgData.faceLandmarks,-ref_box);

if (polyarea(mouth_poly(:,1),mouth_poly(:,2)) < 1)
    mouth_center = mean(mouth_poly,1);
    mouth_poly = box2Pts(inflatebbox([mouth_center mouth_center],3,'both',true));
end
% get the convex hull of the mouth polygon for this purpose
kk = convhull(mouth_poly+rand(size(mouth_poly))*.01);
mouth_poly = mouth_poly(kk,:);

if (isfield(imgData,'face_mask'))
    face_mask = imgData.face_mask;
else
    face_mask = poly2mask2(face_poly,size2(I));
end
mouth_mask = poly2mask2(mouth_poly,size2(I));
mouth_mask = mouth_mask & face_mask;
% mouth_mask = imdilate(mouth_mask,ones(3));


% calculate the occlusion pattern for both face_maks

if (isempty(regions))
    [regions,regionOvp,G] = getRegions(conf,currentID,false);
    regionTypes = {}; regionTypes{1} = ones(length(regions),1);
    if useGCRegions
        regions_gc = getGCRegions(conf,imgData);regionTypes{2} = col([2*ones(length(regions_gc),1)]);% 3*ones(length(regions_gc),1)]);
        regions = [regions(:);regions_gc(:)];
    end
    regionTypes = cat(1,regionTypes{:});
else
    regionTypes = ones(length(regions),1);
end

% [r,ovp] = findCoveringSegment(orig_regions,regions_bu(17));
% profile on;
region_sel = 1:length(regions);
roiMask = zeros(size2(I));
roiMask(roi(2):roi(4),roi(1):roi(3)) = 1;
%cropToRoi = @(x) x(roi(2):roi(4),roi(1):roi(3));
cropToRoi = @(x) x & roiMask;
regions = cellfun2(cropToRoi,regions);
toKeep = cellfun(@(x) any(x(:)),regions);
% G = G(toKeep,toKeep);
regions = regions(toKeep);regionTypes = regionTypes(toKeep);

if (strcmp(conf.occlusion.whatFace,'seg'))
    % find best segment approximating face, instead of using landmarks. Do
    [ovp,ints,uns] = regionsOverlap2(regions,face_mask);
    % more than 0 ovp
    [b,ibest] = max(ovp);
    
    % for the best region, remove all pixels which are too far away...
    face_mask_new = regions{ibest};
    bw_old = bwdist(face_mask);
    face_box = pts2Box(face_poly);
    face_scale = BoxSize(face_box);
    %figure,imagesc(face_mask_new.*bw_old)
    face_mask = (bw_old <= face_scale*.05) & face_mask_new;
end

region_sel = region_sel(toKeep);
[regions,toKeep] = removeDuplicateRegions(regions);
region_sel = region_sel(toKeep);regionTypes = regionTypes(toKeep);
% G = G(toKeep,toKeep);
face_mask = cropToRoi(face_mask);
mouth_mask = cropToRoi(mouth_mask);
regions = fillRegionGaps(regions);
[ovp,ints,uns] = regionsOverlap2(regions,face_mask);
toKeep = find(ovp > 0);
ovp = ovp(toKeep); ints = ints(toKeep); uns = uns(toKeep);
regions = regions(toKeep); regionTypes = regionTypes(toKeep);
% G = G(toKeep,toKeep);
region_sel = region_sel(toKeep);

if(toExpand) % TODO - this is uncompatible with region type 2 and will probably crash...
    [regionGroups] = expandRegions(regions,toExpand,[],G);
    % avoid regions for which no element of the group overlaps the face.
    for iGroup = 1:length(regionGroups)
        curGroup = regionGroups{iGroup};
        goods = max(ovp(curGroup),[],2) > 0;
        regionGroups{iGroup} = curGroup(goods,:);
    end
    [regionGroups,regions] = expandRegions(regions,toExpand,regionGroups,G);
    regions = removeDuplicateRegions(regions);
    % profile off;
    [ovp,ints,uns] = regionsOverlap2(regions,face_mask);
end

mouthDist = bwdist(mouth_mask);
faceDist = bwdist(face_mask);
face_area = nnz(face_mask);


regions_shrink = cellfun2(@(x) imerode(x,ones(3)),regions);
face_mask_shrink = imerode(face_mask,ones(3));
[ovp_strict,ints_strict,uns_strict] = regionsOverlap2(regions_shrink,face_mask_shrink);
% figure
areas = col(cellfun(@nnz,regions));
seg_in_face = ints./areas;
face_in_seg = ints/face_area;
areas_rel_face = areas/face_area;
[ovp,ints,uns] = regionsOverlap(regions,mouth_mask);
face_area = nnz(mouth_mask);
seg_in_mouth = ints./areas;
mouth_in_seg = ints/face_area;
areas_rel_mouth = areas/face_area;

min_dist_to_mouth = cellfun(@(x) min(mouthDist(x(:))),regions);
max_dist_to_mouth = cellfun(@(x) max(mouthDist(x(:))),regions);
min_dist_to_face = cellfun(@(x) min(faceDist(x(:))),regions);
max_dist_to_face = cellfun(@(x) max(faceDist(x(:))),regions);
occlusionPatterns = struct;

[angularCoverage] = calcAngularCoverage(face_poly,face_mask,regions,false);
[distCoverage] = calcDistCoverage(face_poly,face_mask,regions,false);



for k = 1:length(regions)
    %     k
    %     mask = regions{k};
    %     x = imResample(double(mask),[7 7],'bilinear');
    %     occlusionPatterns(k).shape = x;
    occlusionPatterns(k).area = areas(k);
    occlusionPatterns(k).seg_in_face = seg_in_face(k);
    occlusionPatterns(k).face_in_seg = face_in_seg(k);
    occlusionPatterns(k).area_rel_face = areas_rel_face(k);
    occlusionPatterns(k).seg_in_mouth = seg_in_mouth(k);
    occlusionPatterns(k).mouth_in_seg = mouth_in_seg(k);
    occlusionPatterns(k).area_rel_mouth = areas_rel_mouth(k);
    occlusionPatterns(k).min_dist_to_mouth = min_dist_to_mouth(k);
    occlusionPatterns(k).max_dist_to_mouth = max_dist_to_mouth(k);
    occlusionPatterns(k).min_dist_to_face = min_dist_to_face(k);
    occlusionPatterns(k).max_dist_to_face = max_dist_to_face(k);
    occlusionPatterns(k).strictlyOccluding = ovp_strict(k)>0;
    occlusionPatterns(k).angular_coverage_min_f = angularCoverage.min_dist_f(:,k);
    occlusionPatterns(k).angular_coverage_max_f = angularCoverage.max_dist_f(:,k);
    occlusionPatterns(k).angular_coverage_min_c = angularCoverage.min_dist_c(:,k);
    occlusionPatterns(k).angular_coverage_max_c = angularCoverage.max_dist_c(:,k);
    occlusionPatterns(k).dist_coverage = distCoverage(:,k);
    occlusionPatterns(k).region_type = regionTypes(k);
    
end

function distCoverage = calcDistCoverage(face_poly,face_mask,regions,toDebug)
% crop around face region to make calculation faster?
inds = sub2ind2(size(face_mask),fliplr(round(face_poly)));
distCoverage = zeros(size(face_poly,1),length(regions));
for k = 1:length(regions)
    bw = bwdist(regions{k});
    distCoverage(:,k) = bw(inds);
end


function angularCoverage = calcAngularCoverage(face_poly,face_mask,regions,toDebug)


% find center of face...
[y,x] = find(face_mask);
face_center = round(mean([x y],1));
% [xx,yy] = meshgrid(1:size(face_mask,2),1:size(face_mask,1));
% xx = xx-face_center(1); yy = yy-face_center(2);
dist_outside = bwdist(face_mask);
dist_inside = bwdist(~face_mask);
% signed distance
dist = dist_outside.*~face_mask-(dist_inside-1).*(face_mask);
%      contour(dist)
% find where each ray intersects the boundary of the image.
%
% 1. rasterise all lines, using facial keypoints
box = [1 size(face_mask,2) 1 size(face_mask,1)];
coords_face_points = {};
for z = 1:size(face_poly,1)
    ray = createRay(face_center,face_poly(z,:));
    [edge_ isInside] = clipRay(ray, box);
    [xx, yy] = LineTwoPnts(ray(1), ray(2), edge_(3), edge_(4));
    coords_face_points{z} = sub2ind2(size(face_mask),[yy(:) xx(:)]);
end

% 2 create rayz regardless of face coordinates system
coords_circle_points = {};
for theta = 0:10:350
    ray = createRay(face_center,pi*theta/180);
    [edge_ isInside] = clipRay(ray, box);
    [xx, yy] = LineTwoPnts(ray(1), ray(2), edge_(3), edge_(4));
    coords_circle_points{end+1} = sub2ind2(size(face_mask),[yy(:) xx(:)]);
end

if (~toDebug)
    [angularCoverage.min_dist_f,angularCoverage.max_dist_f] = ...
        findRayIntersection(coords_face_points,regions,dist);
    [angularCoverage.min_dist_c,angularCoverage.max_dist_c] = ...
        findRayIntersection(coords_circle_points,regions,dist);
else
    [angularCoverage.min_dist_f,angularCoverage.max_dist_f] = ...
        findRayIntersection(coords_face_points,regions,dist,face_mask);
    [angularCoverage.min_dist_c,angularCoverage.max_dist_c] = ...
        findRayIntersection(coords_circle_points,regions,dist,face_mask);
end

function [min_dist,max_dist] = findRayIntersection(coords,regions,dist,face_mask)
min_dist = nan(length(coords),length(regions));
max_dist = -nan(length(coords),length(regions));
% clf;



for r = 1:length(regions)
    %     r = 10
    if (nargin == 4)
        clf; imagesc(regions{r}+face_mask); hold on;axis image;
    end
    
    % plot near each coordinate its number
    if (nargin == 4)
        for z = 1:length(coords)
            
        end
    end
    
    
    for z = 1:length(coords)
        %         z
        c_ = coords{z};
        t_min = find(regions{r}(c_),1,'first');
        if (any(t_min))
            min_dist(z,r) = dist(c_(t_min));
        end
        t_max = find(regions{r}(c_),1,'last');
        if (any(t_max))
            max_dist(z,r) = dist(c_(t_max));
        end
        if (nargin == 4)
            if (any(t_min))
                [yy,xx] = ind2sub(size(face_mask),c_);
                plot(xx,yy,'r--','LineWidth',2);
                [y,x] = ind2sub(size(face_mask),c_(t_min));
                plot(x,y,'d','MarkerSize',7,'MarkerFaceColor','r','MarkerEdgeColor','k');
                [y,x] = ind2sub(size(face_mask),c_(t_max));
                plot(x,y,'o','MarkerSize',7,'MarkerFaceColor','y','MarkerEdgeColor','k');
                %                 drawnow
                
            end
        end
        %             break
    end
    %     find(~isnan(min_dist(:,r)))
    zxc = 1;
end

% displayRegions(im,regions);
%      logarr = logsample(double(r), 1, 100, face_center(1), face_center(2), 50, 36);
%      arr = logsampback(logarr, 1, max(dist(r(:))));
%      logarr = logsample(I, 1, max(dist(r(:))), face_center(1), face_center(2), 10, 36);
% end

% dataMatrix = [cat(2,occlusionPatterns.area);...
%     cat(2,occlusionPatterns.seg_in_face);...
%     cat(2,occlusionPatterns.face_in_seg);...
%     cat(2,occlusionPatterns.area_rel_face);...
%     cat(2,occlusionPatterns.seg_in_mouth);...
%     cat(2,occlusionPatterns.mouth_in_seg);...
%     cat(2,occlusionPatterns.area_rel_mouth)];%;...
%cat(2,occlusionPatterns.shape)];
