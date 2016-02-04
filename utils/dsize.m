function res = dsize(X,dim)
%res = dsize(X,dim) Returns size along the dimensions specified in dim. 
% if dim is ommited defaults to behaviour of size (i.e, all dimensions are
% returned).
if (nargin < 2)
    res = size(X);
    return;
end
if (isscalar(dim))
    res = size(X,dim);
    return;
end

res = zeros(size(dim));
for k = 1:length(dim)
    res(k) = size(X,dim(k));
end
end

