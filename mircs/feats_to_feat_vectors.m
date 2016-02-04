function [vecs,labels] = feats_to_feat_vectors(feats)
% feats = feats_train;




labels = [feats.class];
face_areas = cat(2,feats.face_area); % 0
boxAreas = cat(2,feats.ref_box_area);
face_colors = cat(2,feats.face_color); % 1:3
face_occs = cat(2,feats.face_occupancy); % 4:28
face_poses = cat(2,feats.face_pose);  % 6
face_scores = cat(2,feats.face_score); % 7
mouth_colors = cat(2,feats.mouth_color);
mouth_areas = cat(2,feats.mouth_area);
mouth_occs = cat(2,feats.mouth_occupancy);
region_colors = cat(2,feats.region_color);
region_occ = cat(2,feats.region_occupancy);
region_areas = cat(2,feats.region_area)./boxAreas;
region_ecc = cat(2,feats.Eccentricity);
region_maj = cat(2,feats.MajorAxisLength)./(boxAreas.^.5);
region_min = cat(2,feats.MinorAxisLength)./(boxAreas.^.5);
region_ori = cat(1,feats.Orientation)';
edgeStrength = cat(2,feats.edgeStrength);
ovp_mouth = cat(2,feats.ovp_mouth);
ints_mouth = cat(2,feats.int_region_mouth)./mouth_areas;
ovp_face = cat(2,feats.ovp_region_face);
ints_face = cat(2,feats.ints_region_face)./face_areas;
ovp_face_not = cat(2,feats.ovp_region_face_not);
ints_not_face = cat(2,feats.ints_region_not_face)./(cat(2,feats.ints_region_face) + cat(2,feats.ints_region_not_face));
landmark_occ = cat(2,feats.landmark_occ);

%featureLengths = [3 25 1 1 3 1 25 3 25 1
%[3 25 1 1 3 25 3 25]
%cumsum([3 25 1 1 3 25 3 25])
        % 1:3       4:28        
vecs = [face_colors;face_occs;face_poses;face_scores;mouth_colors;mouth_occs;region_colors;region_occ;...
    region_areas;region_ecc;region_maj;region_min;region_ori;edgeStrength;...
    ovp_mouth;ints_mouth;ovp_face;ints_face;ovp_face_not;ints_not_face;landmark_occ];
% vecs = sparse(double(vecs));
% vecs = [vecs;vl_homkermap(cat(2,feats.region_bow), 1, 'kchi2', 'gamma', 1)];
% vecs = [vecs;vl_homkermap(cat(2,feats.mouth_bow), 1, 'kchi2', 'gamma', 1)];
vecs = [vecs;cat(2,feats.region_bow)];
% vecs = [vecs;cat(2,feats.mouth_bow)];

end