function bb = pts2Box(pts,pts2)
if(nargin ==2)
    pts = [pts pts2];
end
xmin = min(pts(:,1));
xmax = max(pts(:,1));
ymin = min(pts(:,2));
ymax = max(pts(:,2));
bb = [xmin ymin xmax ymax];
end