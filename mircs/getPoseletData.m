function [rects,rects_poselets,poselet_centers,poselet_ids,s,is] = ...
    getPoseletData(conf,curID,xmin,ymin,xmax,ymax)

im1.image_file{1} = getImagePath(conf,curID);
resFile = fullfile('poselets_quick',strrep(curID,'.jpg','_poselets_quick.mat'));
load(resFile);

rects = bounds_predictions.bounds';
rects(:,[3 4]) = rects(:,[3 4])+rects(:,[1 2]);
% rects = rects(:,[2 1 4 3]);
% plotBoxes2(rects(2,[2 1 4 3]),'g');
ovpScores = boxesOverlap(rects,[xmin ymin xmax ymax]);

% choose prediction using weighted score + overlap score
w =.01;
wScore =  double(ovpScores>.5)+w*bounds_predictions.score;
[s,is] = sort(wScore,'descend');
b = bounds_predictions.select(is(1));

%%%selected_poselet_hits = poselet_hits.select(bounds_predictions.src_idx{is(1)});
selected_poselet_hits = poselet_hits;
rects_poselets = selected_poselet_hits.bounds';
rects_poselets(:,[3 4]) = rects_poselets(:,[3 4])+rects_poselets(:,[1 2]);
poselet_ids = selected_poselet_hits.poselet_id;
poselet_centers = boxCenters(rects_poselets);
end