function [nextmls, epts] = ComputeAllNextMLs(mlnet, mainlines, depth)

% Pre-computes the next main-lines,
% up to the provided depth, for each main-line in the
% main-lines network mlnet. Both directions are considered.
% Useful to prepare lookup-tables for speeding up
% algorithms that need to walk through the network.
%
% Output:
% nextmls(mlix, dir).mls = indices of main-lines next to mlix in direction dir (up to provided depth)
%                   .epts = their connection endpoints
%                   .depths = their depths
% epts(dir, mlix1, mlix2) = endpoint (0 or 1, for 1 or 2) where mlix2 is connected to mlix1, at mlix1's dir endpt
%

% initialize datastructure
N = length(mlnet);
if N == 0
  nextmls = []; epts = [];
  return;
end
nextmls(N,2).mls = [];
nextmls(N,2).epts = [];
nextmls(N,2).depths = [];

% Fill it up
epts(2,N,N) = 0;  % allocate mem 
for mlix = 1:length(mlnet)
  for dir = [2 1]
     [nextmls(mlix,dir).mls nextmls(mlix,dir).epts nextmls(mlix,dir).depths] = NextMLs(mlix, mlnet, dir, zeros(1,length(mlnet)), depth);
     % fix connecting endpoints according to nearest-endpt rule
     e = MLEndpoint(mainlines(:,mlix), dir);
     ix = 0;
     for t = nextmls(mlix,dir).mls
       ix = ix+1;
       nextmls(mlix,dir).epts(ix) = ClosestMLEndpoint(mainlines(:,t), e);
       epts(dir, mlix, t) = nextmls(mlix,dir).epts(ix)-1;
     end
  end
end
