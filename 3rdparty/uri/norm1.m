function [out] = norm1(in, dim)

if (nargin < 2)
    dim = 1;
end

out = bsxfun(@rdivide, in, fixdiv(sum(abs(in), dim)));