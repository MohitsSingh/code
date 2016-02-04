function dr = DiagRange(obj)

% Diagonal of BB of all edgels of obj
%

eds = [];
for ecix = 1:length(obj.ecs)
  eds = [eds obj.ecs(ecix).chain];
end

if isempty(eds)
  dr = [];
else
  BB = boundrect(eds');
  dr = sqrt((BB(2,1)-BB(1,1))^2+(BB(2,2)-BB(1,2))^2);
end
