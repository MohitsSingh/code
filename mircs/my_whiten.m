function X = whiten(X,hog_gauss_struct)
d = hog_gauss_struct.d;
lambda_ = .1;
for k = 1:size(X,2)
    X(:,k) = (d./(d.^2+lambda_.^2)).*X(:,k);
end