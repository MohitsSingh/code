function [ boxes ] = sampleOnPerimeter(mask,R)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
[yy,xx] = find(bwperim(mask));
boxes = inflatebbox([xx yy],R,'both',true);

end
