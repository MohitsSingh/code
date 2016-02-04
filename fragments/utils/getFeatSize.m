function [ histSize ] = getFeatSize( globalOpts )
%GETFEATSIZE Summary of this function goes here
%   Detailed explanation goes here
dummy_hist = buildSpatialHist(1,...
    [2;2;0;0],[1 1 3 3], globalOpts);

histSize = length(dummy_hist);

end

