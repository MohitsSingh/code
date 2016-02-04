function S = ComputeAllSidednessOrient(mlabc, real_valued)

% Compute sidedness-orient constrs for every pair
% of main-lines in mlabc(:,ml).
% 
% Output:
% S(a, b, c) = side of main-line b wrt orientation-line built on a;
%              c = constraint type ->
%              c=1 -> orient-line is support line of a;
%              c=2 -> orient-line is perpendicular to a (still passing through center)
%
% If real_valued = true -> S(a, b, c) = signed distance
%


if nargin < 2
  real_valued = false;
end

% constraint type 1: orient-line is support line
N = size(mlabc, 2);
S1 = mlabc(3:5,:)' * [mlabc(1:2,:); ones(1, N)];
S1(abs(S1)<5) = 0;
if not(real_valued)
  S1 = sign(S1);
end

% constraint type 2: orient-line is perpendicular to support line
% construct perpendicular lines
%perp_dirs = mlabc(3:4,:);  % 'up' direction, when orient-line dir is 'right' (like a usual coord system)
pmlabc = zeros(5,N);
for mlix = 1:N
   ml = mlabc(:,mlix);
   ef = [ml(1)+ml(3) ml(2)+ml(4) 1];  % front endpoint
   eb = [ml(1)-ml(3) ml(2)-ml(4) 1];  % back  endpoint
   l = cross(ef,eb)';                 % line from ef to eb
   k = sqrt(l(1)^2+l(2)^2);
   l = l/k;                           % normalize coefficients so that l*p = Euclidean distance of p to line l
   pmlabc(:,mlix) = [ml(1:2); l];
end
% compute constraints
S2 = pmlabc(3:5,:)' * [pmlabc(1:2,:); ones(1, N)];
S2(abs(S2)<5) = 0;
if not(real_valued)
  S2 = sign(S2);
end

% assemble final output
S(:,:,1) = S1;
S(:,:,2) = S2;
