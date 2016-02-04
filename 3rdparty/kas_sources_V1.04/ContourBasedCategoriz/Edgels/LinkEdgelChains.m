function links = LinkEdgelChains(ecs, becs, tan_cont)

% Finds links among edgelchains ecs.
% according to the linking criteria on notes 12.11.
% If the additional constraint tan_cont is set (=true), then the tangent continuity
% between the endpoint of a chain and a point on the other chain must be preserved.
% The chains are binned so that if c is in becs.B(i,j).ecix, then ecs(c) intersects bin i,j.
% The bins are arranged on a rectangular grid, and their locations
% can be reconstructed using becs.range = [minx maxx miny maxy] = range of area they cover.
%
% Input:
% ecs(c).chain(:,n) = [x;y] coordinates of nth pixel of cth chain
% becs = see above
% 
% Output:
% links(:,l) = [c1; c2; endpt1; endpt2; link_pt] =
%              chain ecs(c1) links to ecs(c2) at point link_pt near its endpont endpt1.
%              if tan_cont is false then
%                link_pt is the median point among the ones of ecs(c2) within the search area
%                defined at endpoint endpt1 (endpt=1=start; endpt=2=end)
%              if tan_cont is true then
%                link_pt is the point of ecs(c2) within search area with tangent orientation most
%                similar to the one at endpoint1 (the median if more than one with same lowest similarity)
%              if an endpt of ecs(c2) is within search area then
%                it is stored in endpt2
%              else
%                endpt2=0 (T-junction)
%

% Parse parameters
if nargin < 3
  tan_cont = false;
end

% Reconstruct binning parameters
r = becs.range;
minx = r(1); maxx = r(2); miny = r(3); maxy = r(4);
nbw = size(becs.B,1);                            % number of bins along image width
nbh = size(becs.B,2);                            % number of bins along image height
bw = (maxx-minx)/nbw;                            % bin width
bh = (maxy-miny)/nbh;                            % bin height

% Link one chain at the time
links = [];
for ecix = 1:length(ecs)

  % Current chain
  ec = ecs(ecix);

  for endpt = [1 2]                              % try both endpts

  % Determine search area A
  % cone-of-search (trapeze) like on notes of 12.11.2004
  [A d] = BuildTrapeze(ec, endpt);

  % Which bins intersect the search area ?
  % assuming that both bins dimensions are larger than both search area's ones, then
  % it suffices to check in which bin each of the area's corners is
  bins = [1+floor((A(1,:)-minx)/bw); 1+floor((A(2,:)-miny)/bh)];
  bins = bins(:, find(bins(1,:)>0 & bins(1,:)<=nbw));
  bins = bins(:, find(bins(2,:)>0 & bins(2,:)<=nbh));  % do not consider parts of A outside the edgelchains' range

  % Which edgelchains pass through those bins ?
  ecib = [];                           % edgel-chains in bins
  for b = bins
    ecib = union(ecib, becs.B(b(1),b(2)).ecix);
  end
  ecib = setdiff(ecib, ecix);

  % Which edgelchains intersect the search area ?
  % A point is inside if it has the same sidedness with respect to the support lines
  % of all 4 sides than the opposite point to each side.
  % This leads to an efficient test, where the first constraint (first side) is checked for all points,
  % then the second constraint is checked only for the points respecting the first, and so on.
  % Build the four constraints
  l = zeros(4,3);
  for t = 1:4
     l(t,:) = cross([A(:,t);1], [A(:,mod(t+1-1,4)+1);1])';
     s(t) = sign(l(t,:) * [A(:,mod(t+2-1,4)+1);1]);         % signs always -1, by construction of A
  end
  % Test edgelchains
  for ecix2 = ecib             % loop over edgel-chains in bins could be avoided as well, but anyway only very few in it
      P = ecs(ecix2).chain;    % all points along the chain
      surv = 1:size(P,2);      % points surviving all tests up to now

      % Apply the four 'inside search area' tests
      for t = 1:4
         tp = (sign(l(t,:)*[P(:,surv);ones(1,length(surv))]) == s(t));
         surv = surv(find(tp));
         if isempty(surv)
           break;
         end
      end

      % Is an endpoint of ecix2 inside search area ? If so, which one ?
      endpt2 = 0;
      if find(surv==1) endpt2=1; end
      if find(surv==size(P,2)) endpt2=2; end
     
      % Apply tangent continuity constraint
      if tan_cont & not(isempty(surv))
         a = atan2(d(2),d(1));
         if a<0 a=a+pi; end
         D = ecs(ecix2).d;          % all directions along the chain
         D = D(:,surv); D = atan2(D(2,:),D(1,:));
         D = D+pi*(D<0);
         tp = abs(a-D); tp = tp.*(tp<=pi/2) + (pi-tp).*(tp>pi/2);
         link_pt = surv(find(tp==min(tp)));    
         link_pt = link_pt(round((1+length(link_pt))/2));  % save the point with most similar tangent as linking point
         surv = surv(find(tp<0.2)); % tangent angle difference tolerance
      end

      if not(tan_cont) & not(isempty(surv))
         % save the median point as the linking point
         link_pt = surv(round((1+length(surv))/2));
      end
    
      % has any point survived all constraints ?
      % if so, cast a link
      if not(isempty(surv))
         links = [links [ecix; ecix2; endpt; endpt2; link_pt]];
      end
  end  % loop over edgelchains in the bins intersecting the search area

  end % for over two endpts

end % loop over edgelchains
