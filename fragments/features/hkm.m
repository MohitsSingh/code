function [ psix ] = hkm( W )
%HKM Summary of this function goes here
%   Detailed explanation goes here
%     psix = vl_homkermap(W, 1, 'kinters', 'gamma', .5);
 psix = vl_homkermap(W, 1, 'kchi2', 'gamma', .5);
% psix = W;
end

