function [ bbox_data] = getClassBoundingBoxes( gt_data, cls)
%GETCLASSBOUNDINGBOXES Summary of this function goes here
%   Detailed explanation goes here

bbox_data = struct('image_ind',{},'bboxes',{},'isDifficult',{});
n = 0;
for t = 1:length(gt_data)
    clsinds = strmatch(cls,{gt_data(t).objects(:).class},'exact');
    if (isempty(clsinds))
        continue;
    end
    n = n+1;
    bbox_data(n).image_ind = t;
    objs = gt_data(t).objects(clsinds);
    bbox_data(n).boxes = cat(1,objs.bbox);
    bbox_data(n).isDifficult = [objs.difficult];
end


end

