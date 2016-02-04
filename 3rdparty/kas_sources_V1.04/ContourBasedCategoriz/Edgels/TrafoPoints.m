function Pt = TrafoPoints(P, T, s, ctr)

% Transform points P(:,i) by translation T and scale s.
% If length(s) == 2 -> apply anisotropic scaling.
% The translation and scale apply relative to ctr (if omitted, take the center of P).
%

if nargin < 4
  ctr = mean(P')';
end

if length(s) == 1
  s = [s; s];
end

Pt = [ (P(1,:)-ctr(1))*s(1) + ctr(1)+T(1);
       (P(2,:)-ctr(2))*s(2) + ctr(2)+T(2) ];
