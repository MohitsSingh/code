function data = normalize_data_cifar(data,cifar_normalization)
if (~iscell(data))
    data = {data};
end

% make sure class of data is correct
if isa(data{1},'uint8')
    data = cellfun2(@single,data);
else
    data = cellfun2(@(x) single(x)*255,data);
end

data = cellfun2(@(x) imResample(x,[32 32],'bilinear'),data);
data = cat(4,data{:});

data = bsxfun(@minus, data, cifar_normalization.dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

% contrastNormalization
nImages = size(data,4);
z = reshape(data,[],nImages) ;
z = bsxfun(@minus, z, mean(z,1));
n = std(z,0,1) ;
z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
% data = reshape(z, 32, 32, 3, []) ;


% z = reshape(data,[],nImages) ;
%   W = z(:,set == 1)*z(:,set == 1)'/nImages ;
%   [V,D] = eig(W) ;
% the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
z = cifar_normalization.whitenMatrix*z ;
data = reshape(z, 32, 32, 3, []) ;
