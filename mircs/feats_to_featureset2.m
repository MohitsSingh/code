function [F,classes,ticklabels] = feats_to_featureset2(feats,dets)

face_areas = cat(2,feats.face_area); % 1
boxAreas = cat(2,feats.ref_box_area);
% face_colors = cat(2,feats.face_color); % 2:4
% face_occs = cat(2,feats.face_occupancy); % 5
face_poses = cat(2,feats.face_pose);  % 6
face_scores = cat(2,feats.face_score); % 7
% mouth_colors = cat(2,feats.mouth_color);
mouth_areas = cat(2,feats.mouth_area);
% mouth_occs = cat(2,feats.mouth_occupancy);
% region_colors = cat(2,feats.region_color);
region_occ = cat(2,feats.region_occupancy);
region_areas = cat(2,feats.region_area);
% region_ecc = cat(2,feats.Eccentricity);
% region_maj = cat(2,feats.MajorAxisLength);
% region_min = cat(2,feats.MinorAxisLength);
% region_ori = cat(1,feats.Orientation)';
edgeStrength = cat(2,feats.edgeStrength);
% ovp_mouth = cat(2,feats.ovp_mouth);
ints_mouth = cat(2,feats.int_region_mouth)./mouth_areas;
% ovp_face = cat(2,feats.ovp_region_face);
ints_face = cat(2,feats.ints_region_face)./face_areas;
% ovp_face_not = cat(2,feats.ovp_region_face_not);
ints_not_face = cat(2,feats.ints_region_not_face)./(cat(2,feats.ints_region_face) + cat(2,feats.ints_region_not_face));

classes = [feats.class];
% imageIndices = [feats.imageIndex];


region_occ = reshape(region_occ,5,5,[]);
region_occ_x = sum(region_occ,2);
region_occ_top = (region_occ_x(1,1,:))>0;
region_occ_horz = max(region_occ_x,[],1) == 5;


%%lipScores = dets.cluster_locs(imageIndices,12); % TODO!
lipScores = zeros(size(feats'));
f1 = abs(face_poses-7)<=3;
f2 = face_scores >= -.5;
f3 = ints_mouth>.5;
%f4 = (ints_not_face./region_areas) > .3;
% f4 = (ints_not_face./[cat(2,feats.ints_region_face) + ints_not_face]);
f4 = ints_not_face;
f5 = ints_face;
f6 = sum(edgeStrength);
f7 = ints_mouth;
f8 = lipScores';
f9 = face_scores;
f10 = region_areas./boxAreas <= .3;
f11 = row(squeeze(region_occ_top));
f12 = row(squeeze(region_occ_horz));


F = [f1;f2;f3;f4;f5;f6;f7;f8;f9;f10;f11;f12];
ticklabels = {'face_poses',...
    'face_scores',...
    'ints_mouth',...
    'ints_not_face',...
    'ints_face',...
    'edge_strength',...
    'ints_m_2',...
    'lip_scores',...
    'face_scores',...
    'area_in_box',...
    'total'};

