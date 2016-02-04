function [ a ] = replaceNanWith(a,x)
%REPLACENANWITH Summary of this function goes here
%   Detailed explanation goes here
    a(isnan(a)) = x;
end

