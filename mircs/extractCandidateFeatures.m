function [I_sub_color,face_mask,regions_sub,subUCM] = extractCandidateFeatures(conf,currentID,faceBoxShifted,lipRectShifted,faceLandmarks,debug_)
candidates = {};
regions_sub = {};
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
facePts = faceLandmarks.xy;
if (isempty(facePts))
    return;
end

box_c = round(boxCenters(lipRectShifted));
sz = faceBoxShifted(3:4)-faceBoxShifted(1:2);
bbox = round(inflatebbox([box_c box_c],floor(sz/1.5),'both',true));
bbox = clip_to_image(bbox,I);
if (any(~inImageBounds(size(I),box2Pts(bbox))))
    return;
end

I_sub_color = I(bbox(2):bbox(4),bbox(1):bbox(3),:);


I_sub = rgb2gray(I_sub_color);

ucmFile = fullfile('/home/amirro/storage/gpb_s40/',strrep(currentID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
ucm = ucm(ymin:ymax,xmin:xmax); %#ok<NODEF>
subUCM = ucm(bbox(2):bbox(4),bbox(1):bbox(3));
E = subUCM;


xy = faceLandmarks.xy;
xy_c = boxCenters(xy);

chull = convhull(xy_c);
% find the occluder!! :-)
c_poly = xy_c(chull,:);
c_poly = bsxfun(@minus,c_poly,bbox(1:2));
face_mask = poly2mask(c_poly(:,1),c_poly(:,2),size(E,1),size(E,2));

[ii,jj,vv] = find(subUCM);
if (length(ii) < 5)
    return;
end
% boundaries = bwbounadaries(face_mask);

E = E>.2;
if (~any(E(:)))
    return
end
E = bwmorph(E,'thin',Inf);
if (nnz(E) < 5)
    return
end

if (debug_)
end

regions_sub = combine_regions_new(subUCM,.1);
regions_sub = fillRegionGaps(regions_sub);
areas = cellfun(@nnz,regions_sub);
regions_sub(areas/numel(E)>.6) = [];

