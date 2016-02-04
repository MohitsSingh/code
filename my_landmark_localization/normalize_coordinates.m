function [ phis ,factors] = normalize_coordinates( phis,Is,single_pt)
%NORMALIZE_COORDINATES Summary of this function goes here
%   Detailed explanation goes here
if (nargin < 3)
    single_pt = false;
end
sizes = cell2mat(cellfun2(@size2,Is));% height/width
factors = sizes(:,1);
if (single_pt)
    phis(:,1:2) = bsxfun(@rdivide,phis(:,1:2),factors);
else
    phis(:,:,1:2) = bsxfun(@rdivide,phis(:,:,1:2),factors);
end
end

