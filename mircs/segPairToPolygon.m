function [xx,u_major,u_minor,xy] = segPairToPolygon(seg1,seg2,centerPoint,scale,debug_,I,reverse)
if (nargin < 7)
    reverse = false;
end
center_y = centerPoint(2);
center_x = centerPoint(1);
xy = reshape([seg1';seg2'],2,[])';
% xy = [seg1;seg2]';
%xy = xy/(conf.straw.dim/s1);
%xy = bsxfun(@plus,xy,face_box(1:2));

xy_c = mean(xy,1);
u_major = segs2vecs([seg1;seg2]);
[u_major] = normalize_vec(u_major,2);
u_major = mean(u_major);
% if (reverse)
%     u_major = -u_major;
% end
u_minor = [-u_major(2) u_major(1)];
% check when the vector meets the y value of the mouth
if (debug_)
    clf,
    subplot(1,2,1);
    imagesc2(I); hold on; plotPolygons(xy_c,'r+','MarkerSize',11,'LineWidth',3);
    quiver(xy_c(1),xy_c(2),10*u_major(1),10*u_major(2),'g');
end
if (u_major(2)~=0)
    alpha_ = (center_y-xy_c(2))/u_major(2);
else
    alpha_ = (center_x-xy_c(1))/u_major(1);
end

p1 = xy_c+alpha_*u_major;


%scale = face_box(4)-face_box(2);
if (isscalar(scale))
    scale = [scale scale];
end

% u_major = u_major*reverse;
% u_minor = u_minor*reverse;
p2 = p1+u_major*(scale(1));
if (debug_)
    hold on;plotPolygons(p1,'md');
    hold on;plotPolygons(p2,'r*');
end
%scale1 = scale;scale2 = scale/2.5;
t1 = u_major*scale(1);t2 = u_minor*scale(2)/2;
x0 = p2+t2;
x1 = x0-t1;
x2 = x1-2*t2;
x3 = x2+t1;
xx = [x1;x2;x3;x0];

if (reverse)
    xx = rotate_pts(xx,180,p1);
end
% % xx = bsxfun(@minus,xx,u_major*scale(1)/5);