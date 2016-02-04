function b = isContinuationOf(line1,line2)
% check if line2 is a continuation of line 1
% edge1 = createEdge(line1);
% edge2 = createEdge(line2);
% assume point are ordered so p21 is near p12.
% n1 = normalize_vec(line1.p2-line1.p1,2);
% n2 = normalize_vec(line2.p2-line2.p1,2);

T = lineAngle(line1,line2);
d = 10;
b = T > 360 - d || T < d;
% make sure that lines are not to far apart on the orthogonal axis.
d = distancePointLine([line2(1:2);line2(3:4)+line2(1:2)],line1);
b = b && max(d) < 5;