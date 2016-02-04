function [seglist,edgelist] = processEdges(E)
seglist = {};
edgelist = {};
[edgelist ~] = edgelink(E, []);
seglist = lineseg(edgelist,1);
end