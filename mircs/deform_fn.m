function [x,y] = deform_fn(sz,x,y,T)
center = sz(1:2)/2;
T(1) = T(1)*sz(2);
T(2) = T(2)*sz(1);
T(3) = 1+T(3);
dxy = T(5:end);
dx = dxy(1:end/2);
dy = dxy(end/2+1:end);
% T(4) = ;
x_c = x-center(2); % to center.
y_c = y-center(1); % to center.
R = rotationMatrix(T(4));
xy = (R*[x_c,y_c]')'; % rotate
% scale
xy = xy*T(3);
x = xy(:,1) + center(2)+T(1)+dx';
y = xy(:,2) + center(1)+T(2)+dy';
