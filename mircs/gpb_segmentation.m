function seg_data = gpb_segmentation(conf,I)

% if (nargin == 3)
%     seg_data = loadOrCalc(conf,@gpb_segmentation_helper,I,cachePath);
% else

[gPb_orient, gPb_thin, textons] = globalPb(I);
gPb_orient = single(gPb_orient);
gPb_thin = single(gPb_thin);
% textons = single(textons);
ucm = contours2ucm(double(gPb_orient));
regions  = combine_regions_new(ucm,.1);
regionOvp = regionsOverlap(regions);
G = regionAdjacencyGraph(regions);
% seg_data.gPb_orient = gPb_orient;
seg_data.gPb_thin = gPb_thin;
% seg_data.textons = textons;
seg_data.ucm = single(ucm);
seg_data.regions = regions;
seg_data.regionOvp = single(regionOvp);
seg_data.G = G;
end