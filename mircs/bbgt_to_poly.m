function [gt_poly,angle] = bbgt_to_poly(objs)
pos0 = [objs.bb objs.ang];
t=pos0(5); c=cosd(t); s=sind(t); R=[c -s; s c];
rs=pos0(3:4)/2; pc=pos0(1:2)+rs;
x0=-rs(1); x1=rs(1); y0=-rs(2); y1=rs(2);
pts=[x0 y0; x1 y0; x1 y1; x0 y1]*R'+pc(ones(4,1),:);
xs=pts(:,1); ys=pts(:,2);
gt_poly = [xs ys];
angle = objs.ang;
end