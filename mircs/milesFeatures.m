function [m,C] = milesFeatures(feats,C)
    if (nargin < 2)
        C = cat(1,feats{:});
    end
    m = zeros(size(C,1),length(feats));
    for k = 1:length(feats)
        k
        m(:,k) = min(l2(feats{k},C))';        
    end    
end