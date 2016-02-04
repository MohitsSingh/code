function [ src,dst ] = propagate_matches( matches,T)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%
pts_dst = matches(1).xy_dst;
im = 1:size(pts_dst,1);
goods = true(size(im));
disp(nnz(goods))
for t = 1:length(matches)-1
    cur_dst = matches(t).xy_dst;
    cur_dst = cur_dst(im,:);
    next_src = matches(t+1).xy_src;
    
    D = l2(cur_dst,next_src);
    [m,im] = min(D,[],2);
    goods(m >T) = false;
    disp(nnz(goods))
end
src = matches(1).xy_src;
dst = matches(end).xy_dst(im,:);
src = src(goods,:);
dst = dst(goods,:);
