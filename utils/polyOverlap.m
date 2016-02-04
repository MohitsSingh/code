function ovp = polyOverlap(p1,p2)

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

a1 = polyarea2(p1);
a2 = polyarea2(p2);
intArea = polyarea2(polybool2('&', p1,p2));
ovp = intArea/(a1+a2-intArea);

end

