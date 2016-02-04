function [ z ] = inMask( m,pts )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
assert(all(inImageBounds(m,pts)));

z = m(sub2ind2(size(m),round(fliplr(pts))));

end

