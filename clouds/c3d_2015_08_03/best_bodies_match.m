function [matches] = best_bodies_match(x1,x2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    toIterate = false;
end
d = l2(double(x1),double(x2));
[~,min_i] = min(d');min_i = min_i';
[~,min_j] = min(d);min_j = min_j';
M = [col(1:length(min_i)) min_i];
matches_diff =  min_j(min_i)-(1:size(min_i))';
matches = M(matches_diff==0,:)';

end

