function [ t] = none( x )
%NONE Check if all elements of x are zero.
%   Detailed explanation goes here
t = ~any(x(:));

end

