function routes = sampleRoutes(G,routeLength,nRoutes,startNode)
if nargin < 2
    nRoutes = 100;
end
if nargin < 4
    startNode = 1;
end
routes = zeros(nRoutes,routeLength);
routes(:,1) = startNode;
n = size(G,1);
% find neighbors of all nodes
[r,ir] = sort(G,2,'descend');
nNeighbors = sum(G,2);
nodes = struct('n',{});
for t = 1:n
   nodes(t).n = find(G(t,:)); 
end
for iNode = 2:routeLength
    prevNode = routes(:,iNode-1);
    for t = 1:nRoutes
        curNeighbors = nodes(prevNode(t)).n;
        routes(t,iNode) = vl_colsubset(curNeighbors,1,'Random');
    end            
end
routes = unique(routes,'rows');

