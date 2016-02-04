function [La, Ld] = ComputeAllLocAnglesDists(mlabc)

% Compute location-angles and location-dists for each pair of main-lines in mlabc
% 
% Output:
% La(a,b) = direction (in [0,2*Pi]) of vector going from ctr of main-line a to ctr of main-line b
% Ld(a,b) = length of vector
%

N = size(mlabc, 2);
P = mlabc(1:2,:);
Px = ones(N,1) * P(1,:);
Py = ones(N,1) * P(2,:);
Dx = Px-Px';   % direction vector
Dy = Py-Py';

% location-angles
La = atan2(Dy,Dx);
La(La<0) = 2*pi+La(La<0);

% location-dists
Ld = sqrt(Dx.*Dx+Dy.*Dy);
Ld(Ld==0) = 1;                % avoid div-by-zero
