function [mls, eds] = ConstructMainLines(ec, verbose)

% Find main lines of edgel chain ec
%
% Output: 
% mls(:,i) = [cx; cy; orient; length] of the ith main line
%

if nargin < 2
  verbose = false;
end

% Curve properties
cv = ec.cv;
t = ec.t;
chain = ec.chain;
cvd = fnder(cv,1);                       % first derivative funct (of t)
cvdd = fnder(cv,2);                      % second derivative funct (of t)
k = ec.k;                                % curvature at every t

% Visual analysis
if verbose
  p = fnval(cv, t(1:5:end));               % some points on the curve
  d = fnval(cvd, t(1:5:end));              % first deriv at those points
  dd = fnval(cvdd, t(1:5:end));            % second deriv at those points

  figure, axis ij, hold on;
  plot(chain(1,:), chain(2,:), 'ob');    % plots edgel chain
  fnplt(cv,'g');                         % plot fitted curve
  quiver(p(1,:),p(2,:),dd(1,:),dd(2,:)); % plot magnitude of second deriv directed towards center of 'curv circle'

  figure, fnplt(cv,'g'), axis ij, hold on;
  fnplt(cv,'g');                         % plot fitted curve
  quiver(p(1,:),p(2,:),d(1,:),d(2,:));   % plot tangent orientation

  figure, plot(k), hold on;              % plot curvature as a fct of t -> sudden peaks corresp to corners !
  plot([0 t(end)], [0 0], 'r');          % plot x-axis, to better see when curvature changes sign
  axis([0 length(t) -0.5 0.5]);          % scale plot to cut corners off
end

% Currently assume open curve
d = ec.d;                                % tangent orientation at every t
[mls eds] = FindMainLines(chain, cv, t, k, d, verbose);
