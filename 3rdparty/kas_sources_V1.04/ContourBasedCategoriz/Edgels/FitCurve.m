function ecs = FitCurve(ecs)

% Fits smoothed splines to each edgelchain ecs(c).chain.
% Output:
% ecs(c).t  = curve length t(i) at point i (pixel)
% ecs(c).cv = the smoothed spline
% ecs(c).d  = first derivative at every t
% ecs(c).k  = curvature at every t
%

for ecix = 1:length(ecs)
  % Current chain
  ec = ecs(ecix).chain;

  % Basic info about ec
  x = ec(1,:); y = ec(2,:);
  xy = [x;y]; dec = diff(xy.').';
  t = cumsum([0, sqrt([1 1]*(dec.*dec))]);    % curve length t(i) at point i (pixel)

  % Curve fitting
  %cv = csaps(t,xy, 0.01);
  cv = spaps(t,xy, 50);                       % smoothed approximation (50 = tolerance)
  err = fnval(cv,t) - ec;
  err = mean(sqrt([1 1]*(err.*err)));

  % Curve properties
  cvd = fnder(cv,1);                          % first derivative funct (of t)
  cvdd = fnder(cv,2);                         % second derivative funct (of t)
  k = Curvature(cvd, cvdd, t);                % curvature at every t
  d = fnval(cvd, t);                          % first derivative at every t
 
  % Format output
  ecs(ecix).t  = t;
  ecs(ecix).cv = cv;
  ecs(ecix).d  = d;
  ecs(ecix).k  = k;
end
