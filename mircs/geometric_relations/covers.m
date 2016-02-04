function b = covers(p1,p2) 
% check length of projection of p1 on p2, bounded by edges of p2 on this
% axis.
line_2 = createEdge(p2.startPoint,p2.endPoint);

[dist pos1] = distancePointEdge(p1.startPoint,line_2);
[dist pos2] = distancePointEdge(p1.endPoint,line_2);
p1 = min([pos1;pos2]);
p2 = max([pos1;pos2]);
p2 = min(1,p2);
p1 = max(0,p1);
b = p2-p1;
