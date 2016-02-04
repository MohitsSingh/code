function [mlix, ept] = NextMLOnChain(cands_eds, curr_ed)

% Returns the main-line mlix among cand_eds which
% comes next after curr_ed on the edgelchain.
% cand_eds must be sorted in ascending edgel order.
%
% If curr_ed has only one edgel, or if both of them have the same value, then
% return the main-line with the closest edgel-endpt

% determine direction
if length(curr_ed) == 1
  dir = 0;
else
  dir = sign(curr_ed(2) - curr_ed(1));
end

mlix = [];
ept = [];
if not(dir==0)
  if dir==-1
    cands_eds = reverse(cands_eds);
  end
  for ed = cands_eds
     if (dir*ed(4) > dir*curr_ed(2) | dir*ed(3) > dir*curr_ed(2)) & ...
        IntervalsOverlap(sort(ed(3:4)),sort(curr_ed))/(abs(ed(4)-ed(3))+1) < 0.5
        mlix = ed(1);
        if dir==1 ept=1; else ept=2; end
	break;
     end
  end
else
  % when dir==0 just return the ml with closest edgel-endpt
  dists_ept1 = abs(cands_eds(3,:)-curr_ed(1));
  dists_ept2 = abs(cands_eds(4,:)-curr_ed(1));
  [best_val best_ept] = min([dists_ept1; dists_ept2]);
  [trash t] = min(best_val);
  mlix = cands_eds(1,t);
  ept = best_ept(t);
end
