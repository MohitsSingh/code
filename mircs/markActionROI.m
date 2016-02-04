function [action_rois,true_ids,train_ids,train_labels] = markActionROI(conf,roiPath)

conf.max_image_size = inf;
conf.get_full_image = true;
if (nargin < 2)
    roiPath = '~/storage/action_rois';
end
annotationDir = fullfile(roiPath,conf.classes{conf.class_subset});

[train_ids,train_labels,all_train_labels] = getImageSet(conf,'train');
%     [test_ids,test_labels,all_test_labels] = getImageSet(conf,'test');
conf.max_image_size = inf;
conf.get_full_image = true;
true_ids = [];
action_rois = selectSamples(conf,train_ids(train_labels),annotationDir);
%     action_rois = selectSamples(conf,test_ids(test_labels),annotationDir);
% % % action_rois = imrect2rect(action_rois);
% % % true_ids = train_ids(train_labels);
% % % for k = 1:round(length(true_ids)/10):length(true_ids)
% % %     clf;imshow(getImage(conf,true_ids{k}));
% % %     hold on;
% % %     plotBoxes2(action_rois(k,[2 1 4 3]));
% % %     pause
% % % end
% %
% %

end
