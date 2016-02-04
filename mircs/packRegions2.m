function Z = packRegions2(regions)
%PACKREGIONS Summary of this function goes here
%   Detailed explanation goes here

% for k = 1:length(regions)
%     regions{k} =  find(regions{k});
% end


Z{1} = zeros(size(regions{1}));
nLayers = 1;
ppp = randperm(length(regions));

for u = 1:length(ppp)
    k = ppp(u);
    curRegion = regions{k};    
    foundFreeSpace = false;
    for iLayer  = 1:nLayers        
        if (~any(Z{iLayer}(curRegion)))
            Z{iLayer}(curRegion) = k;
            foundFreeSpace = true;
            break;
        end
    end
    if (~foundFreeSpace)
        nLayers = nLayers+1;
        Z{nLayers} = k*curRegion;
    end
end

% % % % % 
% % % % % Z{1} = zeros(size(regions{1}));
% % % % % curLayer = 1;
% % % % % for k = 1:length(regions)
% % % % %     if (any(Z{curLayer}(regions{k})))
% % % % %         curLayer = curLayer+1;
% % % % %         Z{curLayer} = zeros(size(regions{1}));
% % % % %     end
% % % % %     Z{curLayer}(regions{k}) = k;
% % % % % end
% % % % % 
