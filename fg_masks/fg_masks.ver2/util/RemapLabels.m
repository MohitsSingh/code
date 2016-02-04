function [L] = RemapLabels(L)
%REMAPLABELS Remaps a label image so range is continuous and start with 1. 
%   Detailed explanation goes here
    uniqueVals = unique(L(:));
    L_t = L;
    for iVal = 1:length(uniqueVals)
        v = uniqueVals(iVal);
        L_t(L == v) = iVal;
    end
    L = L_t;
end