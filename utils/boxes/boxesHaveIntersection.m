function hasIntersection = boxesHaveIntersection(bb)

x_intersection = intersect_intervals2(bb(:,[1 3]));
y_intersection = intersect_intervals2(bb(:,[2 4]));
x_intersection = max(x_intersection,x_intersection');
y_intersection = max(y_intersection,y_intersection');
hasIntersection = x_intersection & y_intersection;

