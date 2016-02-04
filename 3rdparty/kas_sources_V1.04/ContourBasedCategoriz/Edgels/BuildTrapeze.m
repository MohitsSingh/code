function [A, d] = BuildTrapeze(ec, endpt)

% Builds a trapeze-like search area A
% at the specified endpoint of edgelchain ec.
% Also returs tangent direction d at endpoint.
% Input:
% endpt == 1 -> first chain point,
% endpt == 2 -> last chain point
% 

if endpt == 1
  pix = 1;
else
  pix = size(ec.chain,2);
end

% endpoint and tangent direction
p = ec.chain(:,pix);         % endpoint's coordinates
d = ec.d(:,pix);             % tangent direction
if endpt == 1
  d = -d;                    % reverse direction, to make it point away from the curve
end
d = d/sqrt(sum(d.*d));       % normalize to length 1

% Search area
% trapeze, with bases 2,4, and height 12. The minor base rests on the endpoint, and
% the axis points away from the curve
%n = [-d(2); d(1)];           % perpendicular direction
%A = [p-2*n/2 p+2*n/2 p+4*n/2+12*d p-4*n/2+12*d];

% UPDATE: 25.2.05
% Search area
% trapeze, with bases 2,6, and height 12. The minor base rests on the endpoint, and
% the axis points away from the curve
n = [-d(2); d(1)];           % perpendicular direction
A = [p-2*n/2 p+2*n/2 p+6*n/2+12*d p-6*n/2+12*d];
