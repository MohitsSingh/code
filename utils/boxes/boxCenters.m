function [ xy ] = boxCenters( boxes )
%BOXCENTERS Summary of this function goes here
%   Detailed explanation goes here
xy = [mean(boxes(:,[1 3]),2),mean(boxes(:,[2 4]),2)];

end

