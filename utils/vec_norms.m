function norms= vec_norms(X,dim,p)
%VEC_NORMS Get vector normals.
if nargin < 2
    dim = 1;
end
if nargin < 3
    p = 2;
end
% warning('dropped 1/p!!!!');
norms = sum(X.^p,dim).^(1/p);
end

