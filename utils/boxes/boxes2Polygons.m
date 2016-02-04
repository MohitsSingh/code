function [all_corners,all_box_inds] = boxes2Polygons(boxes)

all_corners = {};
all_box_inds = {};
for iBox = 1:size(boxes,1)
    all_corners{iBox} = box2Pts(boxes(iBox,:));
    all_box_inds{iBox} = iBox*ones(4,1);
end