function region_sel = filterOccluders(occluders)
region_sel = occluders.region_sel;
occlusionPatterns = occluders.occlusionPatterns;
t = find(region_sel);

region_sel(t) = region_sel(t) & [occlusionPatterns.seg_in_face] >0 & ...
    [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < 1 &...
    [occlusionPatterns.seg_in_mouth] > 0 & [occlusionPatterns.area_rel_face] < 2;

end