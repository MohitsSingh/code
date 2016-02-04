function [bb_pos,bb_neg] = sampleUsingRegion(boxes,mask)%,min_pos_ovp,max_neg_ovp)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

[~,~,A] = BoxSize(boxes);
sums = sum_boxes(double(mask),boxes);
sums = sums./A;
boxes = [boxes sums];
pos_keep = nms(boxes,.7);
pos_keep = pos_keep(1:min(3,length(pos_keep)));
bb_pos = boxes(pos_keep,:);
boxes(pos_keep,:) = [];
b12 = max(boxesOverlap(boxes,bb_pos),[],2);
boxes(b12>.3,:) = [];
boxes(:,end) = 0;
bb_neg = boxes(nms(boxes,.7),:);

end

