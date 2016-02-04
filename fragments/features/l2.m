function d = l2(A,B)
% calcualtes the squared L2 distance
% A - NxD
% B - MxD
d = bsxfun(@plus, sum(B.^2,2)', bsxfun(@plus, sum(A.^2,2), -2*A*B'));
