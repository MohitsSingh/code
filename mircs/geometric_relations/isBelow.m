function b = isBelow(p2,p1)
b = sum((p2.xy(:,2)-p1.center(2)) > 0)/length(p2.xy(:,2));