function [ src_new,dst_new ] = propagate_through(dst1,dst2,src1,src2)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

D = l2(src1,src2);
[m,im] = min(D,[],2);
T = .5;
goods = m < T;
src_new = dst1(goods,:);
dst_new = dst2(im(goods),:);
end

