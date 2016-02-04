function BB = GetObjBB(obj)

% comment to be written

all_eds = [];
for ecix = 1:length(obj.ecs)
  all_eds = [all_eds obj.ecs(ecix).chain];
end

if isempty(all_eds)
  BB = [];
else
  BB = boundrect(all_eds')';
end
