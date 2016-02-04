function d = l2(A,B)
% calcualtes the squared L2 distance
% A - NxD
% B - MxD
if (isempty(B))
    d = inf(size(A,1),1);
    return;
end
d = bsxfun(@plus, sum(B.^2,2)', bsxfun(@plus, sum(A.^2,2), -2*A*B'));