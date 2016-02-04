function [ out ] = restrictTop( detections, restriction_bb )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    boxTops = [(detections(:,1)+detections(:,3))/2,detections(:,2)];    
    out = inBox(restriction_bb,boxTops);

end

