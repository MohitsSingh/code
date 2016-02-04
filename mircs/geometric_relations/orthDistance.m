function [b1 b2] = orthDistance(p1,p2)
% measure the distance of the two projections of the endpoints 
% of p1 onto p2
d = distancePointLine([line2(1:2);line2(3:4)+line2(1:2)],line1);
line2 = createLine