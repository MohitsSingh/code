function [ I ] = boundary_func( I )
%BOUNDARY_FUNC Summary of this function goes here
%   Detailed explanation goes here
    level = graythresh(I);
    I = I > level;

end

