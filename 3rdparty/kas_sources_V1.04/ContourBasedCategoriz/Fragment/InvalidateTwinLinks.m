function valid = InvalidateTwinLinks(links)

% If two links i,j link the same two edgelchains at the same endpoints, only the first in
% the list 'links' is kept valid (valid(i)=1); the other is set to invalid (valid(j)=0)
%

if isempty(links)
  valid = [];
  return;
end

% valid is set to 1 if the link is valid
valid = ones(1,size(links,2));

% danger is set to 1 if there's any endpt-to-endpt link between two chains
d = max([max(links(1,:)) max(links(2,:))]);
danger = zeros(d,d);

for lix = 1:size(links,2)
  l = links(:,lix);
  
  % is it a valid endpt-to-endpt link ?
  if l(3)>0 & l(4)>0 & valid(lix)
    if danger(l(2),l(1))
      % There is already some link between chains l(1) and l(2)
      % check if at the same endpoints
      dlix = find((links(1,:)==l(2)) & (links(2,:)==l(1)));             % dangerous links indexes
      dlix = dlix(find(links(3,dlix)==l(4) & links(4,dlix)==l(3)));
      if not(isempty(dlix))
         valid(lix) = 0;
      end
    end
    danger(l(1),l(2)) = 1;   
  end
  
end
