function [x1_res,x2_res] = best_bodies_match_2(x1,x2,toIterate)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if nargin < 3
    toIterate = false;
end
d = l2(double(x1),double(x2));
[~,min_j] = min(d);min_j = min_j';
M = [col(1:length(min_i)) min_i];
matches_diff =  min_j(min_i)-(1:size(min_i))';
matches = M(matches_diff==0,:)';
x1_res = x1(matches(:,1),:);
x2_res = x2(matches(:,2),:);
if toIterate
    n = size(x1,1);
    T = M(matches_diff~=0,:)';
    x1 = x1(T(:,1),:);
    x2 = x2(T(:,1),:);
    [x1,x2] = best_bodies_match_2(x1,x2,false);
    
else
    all_matches = matches;
    
end

