function pts = rect2pts(r)
xmin = r(1);
xmax = r(3);
ymin = r(2);
ymax = r(4);
pts = [xmin ymin;...
    xmax ymin;...
    xmax ymax;...
    xmin ymax];

end