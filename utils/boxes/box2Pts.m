function pts = box2Pts(bb)
xmin = bb(1);
xmax = bb(3);
ymin = bb(2);
ymax = bb(4);
pts = [xmin xmax xmax xmin;ymin ymin ymax ymax]';
end