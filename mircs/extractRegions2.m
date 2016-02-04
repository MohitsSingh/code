function [regions,groups,bbox,face_mask,mouth_mask,I,params,region_sel] = extractRegions2(conf,imgData,params)
regions = [];
groups = [];
bbox = [];
face_mask = [];
mouth_mask = [];
I = [];
region_sel = [];
if (imgData.faceScore==-1000)
    return;
end
if (nargin < 3)
    params.inflationFactor = 1.1;
    params.regionExpansion = 2;
    params.ucmThresh = .15;
end
currentID = imgData.imageID;
[I,xmin,xmax,ymin,ymax] = getImage(conf,currentID);
ref_box = [0 0];

if (conf.get_full_image)
    ref_box = [xmin ymin xmax ymax];
end
[xy_c,mouth_poly,face_poly] = getShiftedLandmarks(imgData.faceLandmarks,-ref_box);
face_mask = poly2mask2(face_poly,size2(I));

[regions,regionOvp,G] = getRegions(conf,currentID,false);
if (~conf.get_full_image)
    regions = cellfun2(@(x) x(ymin:ymax,xmin:xmax), regions);
end

region_sel = cellfun(@(x) nnz(x)>0,regions);
ovp = boxRegionOverlap(face_mask,regions(region_sel));
region_sel(region_sel) = region_sel(region_sel) & ovp > 0;
%region_sel = region_sel & ovp > 0;
%regions(ovp==0) = [];
bbox = round(imgData.faceBox);
bbox = round(inflatebbox(bbox,params.inflationFactor*[1 1],'both',false));
mouthBoxPoly = pts2Box(box2Pts(mouth_poly));
mouth_mask = poly2mask2(mouthBoxPoly,size2(I));
mouth_mask = mouth_mask & face_mask; % don't allow mouth to be out of face :-)
G_sub = G(region_sel,region_sel);
[groups] = expandRegions(regions(region_sel),params.regionExpansion,[],G_sub);
