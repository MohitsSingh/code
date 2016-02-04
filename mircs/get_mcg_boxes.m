function [boxes,I] = get_mcg_boxes(conf,curImageData,roiParams)
[rois,subRect,I,scaleFactor] = get_rois_fra(conf,curImageData,roiParams);
curFeatsPath = j2m('~/storage/s40_fra_pred_feats',curImageData.imageID);
mcgPath = j2m('/home/amirro/storage/s40_seg_new',curImageData.imageID);
load(mcgPath,'res');
boxes = res.cadidates.bboxes(:,[2 1 4 3]);
[boxes,uniqueIDX] = BoxRemoveDuplicates(boxes);
% add the selective search boxes!
load(j2m('~/storage/s40_fra_selective_search',curImageData));
boxes = [boxes;res.boxes];
[boxes,uniqueIDX] = BoxRemoveDuplicates(boxes);

% boxScores = res.cadidates.scores(uniqueIDX);
[~,~,orig_areas] = BoxSize(boxes);
intersection = BoxIntersection(boxes, subRect);
[~,~,areas] = BoxSize(intersection);
rel_areas = areas./orig_areas;
boxes = boxes(rel_areas>=1,:);
% boxScores = boxScores(rel_areas>=.9);
boxes = clip_to_image(boxes,subRect);
%     size(I)
%     max(boxes(:,3))
boxes =  bsxfun(@minus,boxes,subRect([1 2 1 2]));
boxes = round(boxes*scaleFactor);
boxes = clip_to_image(boxes,I);