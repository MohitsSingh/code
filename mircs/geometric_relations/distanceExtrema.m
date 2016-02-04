function [b1 b2] = distanceExtrema(p1,p2)
x1 = p1.xy;
x2 = p2.xy;
d = (l2(x1,x2)).^.5;
b1 = min(d(:));
b2 = max(d(:));