function [L] = RemapLabels(L,strict)
%REMAPLABELS Remaps a label image so range is continuous and start with 1.
%   Detailed explanation goes here
uniqueVals = unique(L(:));
L_t = zeros(size(L));
if (nargin == 2 && strict)
    q = 0;
    for iVal = 1:length(uniqueVals)
        v = uniqueVals(iVal);
        props = regionprops(L==v,'PixelIdxList');
        for k = 1:length(props)
            q = q+1;
            L_t(props(k).PixelIdxList) = q;
        end                
    end
    L = L_t;
    return;
end
L_t = L;
for iVal = 1:length(uniqueVals)
    v = uniqueVals(iVal);
    L_t(L == v) = iVal;
end
L = L_t;
end