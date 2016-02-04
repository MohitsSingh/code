function sz = size2(A,flip)
% Returns the size of the first two dimensions of A. Equivalent to
% dsize(A,1:2) or [size(A,1) size(A,2)]. IF flip is specified and true,
% returns the number of columns first, i.e, fliplr(size2(A))
if nargin == 2 && flip
    sz = dsize(A,[2 1]);
else
    sz = dsize(A,1:2);
end
end