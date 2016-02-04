function edgels = GetAllEdgels(obj, orient)

% All edgels on all edgel-chains obj.ecs(:).chain
% if orient -> compute edgel orients too
%

if nargin < 2
  orient = false;
end

edgels = [];
for c = 1:length(obj.ecs)
  add = [ones(1,length(obj.ecs(c).chain))*c; obj.ecs(c).chain];
  if orient
    d = obj.ecs(c).d;
    add = [add; atan2(d(2,:),d(1,:))];
  end
  edgels = [edgels add];
end
