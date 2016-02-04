function [sel0,sel1] = splitSet(setLength,splitRatio)
%SPLITSET Summary of this function goes here
%   Detailed explanation goes here
if (iscell(setLength))
    p = setLength;
    setLength = length(p);
end
sel0 = 1:round(setLength*splitRatio);
sel1 = setdiff(1:setLength,sel0);
if (exist('p','var'))
    sel0 = p(sel0);
    sel1 = p(sel1);
end

end

