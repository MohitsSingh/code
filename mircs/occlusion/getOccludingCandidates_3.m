function [regions,occlusionPatterns,region_scores] = getOccludingCandidates_3(conf,I,curImageData)
regions = curImageData.occlusionPattern.regions;
face_mask = curImageData.occlusionPattern.face_mask; % just for visualization
occlusionPatterns = curImageData.occlusionPattern.occlusionPatterns;
seg_in_face = [occlusionPatterns.seg_in_face];
face_in_seg = [occlusionPatterns.face_in_seg];
dist_coverage = [occlusionPatterns.dist_coverage];
touch_coverage = sum(dist_coverage < 3)/size(dist_coverage,1);
[f1,f2] = BoxSize(curImageData.faceBox);
face_scale = (f1+f2)/2;
mouth_dist_n = [occlusionPatterns.min_dist_to_mouth]/face_scale;
region_sel = mouth_dist_n <= .15;
pixCount = row(cellfun(@(x) nnz(x(face_mask)),regions));
region_type = [occlusionPatterns.region_type];

region_sel =    region_sel & ...
    seg_in_face < .85 &...%% was .85
    face_in_seg < .4 &...
    seg_in_face > .1 &...
    pixCount > 15 &...
    region_type == 1 & touch_coverage < .3;  
regions = regions(region_sel);
meanColors = cellfun2(@(x) meanSegColors(x,I,false),regions);
meanColors = cat(1,meanColors{:});
occlusionPatterns = occlusionPatterns(region_sel);
if (isempty(regions))
    region_scores = [];
    return;
end
seg_in_face = [occlusionPatterns.seg_in_face];
face_in_seg = [occlusionPatterns.face_in_seg];
mouth_dist_n = [occlusionPatterns.min_dist_to_mouth]/face_scale;
% score according to image support
[ucm,gPb_thin] = loadUCM(conf,curImageData.imageID); %#ok<*STOUT>
% further reduce to regions with > 0 pixels in face_mask_smaller

nRegions = length(regions);
ucm_strength_in = zeros(1,nRegions);
gpb_strength_in = zeros(1,nRegions);
ucm_strength_out = zeros(1,nRegions);
gpb_strength_out = zeros(1,nRegions);
above_face = false(1,nRegions);

color_diffs = zeros(1,nRegions);
face_perim_in_segment = zeros(1,nRegions); % ratio of face perimeter touched by segment
seg_perim_in_face = zeros(1,nRegions); % ratio of segment perimeter touched by face
ucm = imdilate(ucm,ones(3));
gPb_thin = imdilate(gPb_thin,ones(3));

% distance distribution from mask to border.
face_perim = bwperim(face_mask);
face_perim_length = nnz(face_perim);
D = bwdist(face_perim);
ds = {};
% find some more properties of the segments:
% 1. ratio of face perimeter covered (small is good)
% 2. color difference between occluding segment and remainder of face
% 3. todo : add T-junctions, etc.

% 4 for each region check if it has any pixel strictly above the face
face_poly = fliplr(bwtraceboundary2(face_mask));
% find upper hull of face_poly (sort by y value)
x = face_poly(:,1);
y = face_poly(:,2);
xs = unique(x);
ys = zeros(size(xs));
for t = 1:length(xs)
    ys(t) = min(y(x==xs(t)));
end

% figure,plotPolygons([x,y],'r--');
% plotPolygons([xs ys],'g-+');

regions_polys = cellfun2(@(x) fliplr(bwtraceboundary2(x)),regions);


for q = 1:nRegions
    curRegion = regions{q};
    curBorders = bwperim(curRegion);
    curBorders_in = curBorders & face_mask;
    curBorders_out = curBorders & ~face_mask;    
    rest_of_face_color = meanSegColors(face_mask & ~curRegion,I,false);
    color_diffs(q) = norm(rest_of_face_color-meanColors(q,:));    
    ds{q} = D(curBorders);
    % find the intersection of the UCM values with the current borders
    ucm_strength_in(q) = mean(ucm(curBorders_in));
    gpb_strength_in(q) = mean(gPb_thin(curBorders_in));
    ucm_strength_out(q) = mean(ucm(curBorders_out));
    gpb_strength_out(q) = mean(gPb_thin(curBorders_out));    
    face_perim_in_segment(q) = nnz(curRegion & face_perim)/face_perim_length;
    seg_perim_in_face(q) = nnz(curBorders & D < 3)/nnz(curBorders);    
    curPoly = regions_polys{q};           
    [is_x,in_f] = ismember(curPoly(:,1),xs);
    % TODO - middle of debugging
    
%     figure,imagesc(I); hold on; plotPolygons(curPoly(is_x,:))
%     is_above = curPoly(is_x,2) < ys(in_f(is_x));
%     plotPolygons(curPoly(is_above,:),'g+');
    above_face(q) = any(curPoly(is_x,2) < ys(in_f(is_x)));
end


ucm_strength_out(isnan(ucm_strength_out)) = 0;
gpb_strength_out(isnan(gpb_strength_out)) = 0;

[M,O] = gradientMag( im2single(I),1);
a=1;
%region_scores =  gpb_strength_in-gpb_strength_out;
region_scores =  ucm_strength_in-ucm_strength_out;
region_scores = region_scores-face_perim_in_segment-seg_perim_in_face+color_diffs/10-seg_in_face;
region_scores(above_face) = -inf;
% sel_ = ~isnan(region_scores);
% occlusionPatterns =occlusionPatterns(sel_);
% for debugging...
region_sel = mouth_dist_n <= .15;

% clf;imagesc2(showRegionPair(face_mask,regions{1}));

% figure,imagesc2(showRegionPair(face_mask,
% moreFun = @(x) disp(nnz(x));
% displayRegions(I,regions,region_scores,[],[]);