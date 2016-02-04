function hasIntersection = boxesHaveIntersection2(boxes1,boxes2)
if nargin < 2
    boxes2 = boxes1;
end
boxes1 = single(boxes1);
boxes2 = single(boxes2);
% find 1d intersections
bb = [boxes1;boxes2];

x_intersection = intersect_intervals2(bb(:,[1 3]));
y_intersection = intersect_intervals2(bb(:,[2 4]));
x_intersection = max(x_intersection,x_intersection');
y_intersection = max(y_intersection,y_intersection');
hasIntersection = x_intersection & y_intersection;




