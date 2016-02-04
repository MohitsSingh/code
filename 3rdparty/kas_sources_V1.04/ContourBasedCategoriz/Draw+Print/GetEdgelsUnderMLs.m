function [edgels, orients] = GetEdgelsUnderMLs(obj, mls, uniq)

% List of edgel coordinates and orientations covered by main-lines mls
%
% if uniq == false -> return also replicated mls
%

if nargin < 3
  uniq = true;
end

if uniq
  mls = unique(mls);
end

edgels = [];
orients = [];
for mlid = mls
  eds = obj.eds(:,mlid);
  bridging = eds(5)>0;
  [new_edgels new_ixs] = GetEdgels(obj.ecs(eds(2)).chain, eds(3), eds(4), not(bridging));
  new_orients = obj.ecs(eds(2)).d(:,new_ixs);             % now orient coded with 2 vals
  new_orients = atan2(new_orients(2,:),new_orients(1,:)); % code orient with one val (stdr format in rest of system)
  edgels = [edgels new_edgels];
  orients = [orients new_orients];
  %
  if eds(5) > 0
    [new_edgels new_ixs] = GetEdgels(obj.ecs(eds(5)).chain, eds(6), eds(7), not(bridging));
    new_orients = obj.ecs(eds(5)).d(:,new_ixs);             % now orient coded with 2 vals
    new_orients = atan2(new_orients(2,:),new_orients(1,:)); % code orient with one val (stdr format in rest of system)
    edgels = [edgels new_edgels];
    orients = [orients new_orients];
  end
end
