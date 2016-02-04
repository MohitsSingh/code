function [X,norms] = normalize_vec(X,dim,p)
%NORMALIZE_VEC Summary of this function goes here
%   Detailed explanation goes here
if (nargin < 2)
    dim = 1;
end
if nargin < 3
    p = 2;
end
%
norms = vec_norms(X,dim,p);
X = bsxfun(@rdivide,X,norms+eps); %X./repmat(norms,size(X,1),1);
end

