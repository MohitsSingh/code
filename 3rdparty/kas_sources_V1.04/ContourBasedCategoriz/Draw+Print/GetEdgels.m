function [edgels, ixs] = GetEdgels(ec, ed1, ed2, cycle)

% Edgels ed1 to ed2 of edgel chain ec
%

if nargin < 4
  cycle = true; 
end

ixs = ed1:ed2;
if (ed1 > ed2) && cycle
  ixs = [ed1:size(ec,2) 1:ed2];  % cycle through endpoint
elseif (ed1 > ed2) && not(cycle)
  ixs = ed1:-1:ed2;
end

edgels = ec(:,ixs);
