function net = RemoveConnections(mlix, conns, net)

% remove all (front/back) connections from conns to mlix
% (corresponding connections from mlix to conns are kept !)
%

for c = conns
  % remove connections from the back of conns
  if not(isempty(net(c).back))
    t = (net(c).back(1,:)==mlix);
    net(c).back = net(c).back(:,not(t));
  end
  %
  % remove connections from the front of conns
  if not(isempty(net(c).front))
    t = (net(c).front(1,:)==mlix);
    net(c).front = net(c).front(:,not(t));
  end
end
