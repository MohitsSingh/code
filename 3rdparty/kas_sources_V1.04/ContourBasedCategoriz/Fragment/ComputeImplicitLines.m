function mlabc = ComputeImplicitLines(mls)

% Compute the coefficients [a b c] of the implicit equation
% ax+by+c=0 of the support line of each main-line mls(:,i),
% and store them in mlabc(:,i)
%
% Output:
%
% mlabc(:,i) = [x; y; a; b; c]
%

mlabc = zeros(5,size(mls,2));
for mlix = 1:size(mls,2)
   ml = mls(:,mlix);
   ef = [ml(3)+cos(ml(5)) ml(4)+sin(ml(5)) 1];  % front endpoint
   eb = [ml(3)-cos(ml(5)) ml(4)-sin(ml(5)) 1];  % back  endpoint
   l = cross(ef,eb)';                           % line from ef to eb
   k = sqrt(l(1)^2+l(2)^2);
   l = l/k;                                     % normalize coefficients so that l*p = Euclidean distance of p to line l
   mlabc(:,mlix) = [ml(3:4); l];
end
