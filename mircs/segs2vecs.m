function [vecs] = segs2vecs(segs)
%SEGS2VECS Summary of this function goes here
%   Detailed explanation goes here
    vecs = segs(:,3:4)-segs(:,1:2);

end

