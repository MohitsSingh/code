function k = Curvature(cvd, cvdd, t)

% Curvature of a function with first
% and second derivatives cvd, cvdd, 
% at all points t
% k = (x'y''-y'x'')/(x'^2+y'^2)^3/2

d = fnval(cvd, t);
dd = fnval(cvdd, t);
k = (d(1,:).*dd(2,:)-d(2,:).*dd(1,:))./(d(1,:).^2+d(2,:).^2).^(3/2);
