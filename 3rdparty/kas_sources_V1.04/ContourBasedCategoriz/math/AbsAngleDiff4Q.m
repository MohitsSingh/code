function d = AbsAngleDiff4Q(a1, a2)

% absolute difference between angles a1 and a2.
% Both a1,a2 are defined on the four quadrans:
% a1 in [0,2pi], a2 in [0,2pi]
% the result d is in [0,pi]

d = abs(a1-a2);
if d>pi 
  d=2*pi-d;
end
