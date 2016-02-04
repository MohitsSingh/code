function [d12,d21,a12,a21] = getNN(X1,X2,k)
%GETNN Returns the k-nearest neighbors of X1 in X2
%   and vice-versa.

% X1 and X2 are p X n and q X n matrices where each row is a data
% point in Rn. k is the number of neighbors to retrieve.
% d12 is an  k X p matrix where d12(i,j) is the j'th neighbor
% of X1(i,:) in X2.
% d21 is an k X q matrix where d21(i,j) is the j'th neighbor
% if X2(i,:) in X1.
% a12 and a21 are matrices with corresponding sizes to d12 and d21
% which specify the locations of the neighbors.
D = l2(X1,X2);
sk = sign(k);
k = abs(k);
% get for each x2 in x2 the nearest elements in x1

if (k == 1)
    [d12,a12] = min(D'*sk,[],1);
    [d21,a21] = min(D*sk,[],1);    
else
    
    d12 = zeros(k,size(X1,1));
    a12 = zeros(k,size(X1,1));
    
    for r = 1:size(D,1)
        [a,ia] = psort(sk*D(r,:)',k);
        d12(:,r) = sk*a;
        a12(:,r) = ia;
    end
    
    d21 = zeros(k,size(X2,1));
    a21 = zeros(k,size(X2,1));
    for r = 1:size(D,2)
        [a,ia] = psort(sk*D(:,r),k);
        d21(:,r) = sk*a;
        a21(:,r) = ia;
    end    
end




