%% regionBasedDetector
% train a

roiPath = '~/storage/cup_rois';
conf.not_crop = true;
[action_rois,true_ids] = markActionROI(conf,roiPath);

% make data for recognition using regions.
