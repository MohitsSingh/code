function region_sel = filterOccluders(occlusionPatterns,model)


region_sel = [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < 1 &...
 [occlusionPatterns.area_rel_face] <= 5;




% % region_sel = [occlusionPatterns.seg_in_face] >0 & ...
% %     [occlusionPatterns.face_in_seg] < .5 & [occlusionPatterns.seg_in_face] < 1;
%     [occlusionPatterns.seg_in_mouth] > 0;
    %[occlusionPatterns.area_rel_face] < 2;% & [occlusionPatterns.seg_in_face] > .1;
% 
end