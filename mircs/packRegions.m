function [regions] = packRegions(regions)
%PACKREGIONS Summary of this function goes here
%   Detailed explanation goes here

for k = 1:length(regions)
    regions{k} =  find(regions{k});
end

% Z{1} = zeros(size(regions{1}));
% curLayer = 1;
% for k = 1:length(regions)
%     if (any(Z{curLayer}(regions{k})))
%         curLayer = curLayer+1;
%         Z{curLayer} = zeros(size(regions{1}));
%     end
%     Z{curLayer}(regions{k}) = k;
% end

