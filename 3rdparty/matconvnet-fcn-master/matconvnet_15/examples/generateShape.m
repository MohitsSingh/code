function z = generateShape(v,baseShapes)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
switch v
    case 1
        f = maskCircle( 0, 2*pi, 10, 20);
        z = padarray(f,[6 6],0,'both');
    case 2
        f = maskCircle( -pi/8, pi/4, 10, 20 );
        z = padarray(f,[6 6],0,'both');
    case 3
        z = zeros(32);
        z(7:26,7:26)=1;
    case 4
        z = zeros(32);
        z(7:26,7:26)=1;
        z = z.*((imrotate(z,45,'bilinear','crop'))>0);
end

