function p = poly2cw2( p )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    x = p(:,1); y = p(:,2);
    [x,y] = poly2cw(x,y);
    p = [x y];

end

