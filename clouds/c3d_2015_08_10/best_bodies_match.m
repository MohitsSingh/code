function [matches,dists] = best_bodies_match(x1,x2,f1,f2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    toIterate = false;
end
d = l2(double(x1),double(x2));
if nargin > 2
    
    % prepare the data matrix....
%     z = [f1f2];
    
%     vl_kdtreebuild(
%     
    L_y = l2(f1(2,:)',f2(2,:)');
    d(L_y>4) = max(d(:))+1;
end

[min_i_d,min_i] = min(d');min_i = min_i';min_i_d = min_i_d';
[min_j_d,min_j] = min(d);min_j = min_j';min_j_d = min_j_d';
M = [col(1:length(min_i)) min_i];
% min_i_d
matches_diff =  min_j(min_i)-(1:size(min_i))';

matches = M(matches_diff==0,:)';
dists = min_i_d(matches_diff==0);
end

