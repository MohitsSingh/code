function plotCircles(circles)
%PLOTCIRCLES Summary of this function goes here
%   Detailed explanation goes here
hold on;
for q = 1:size(circles,2)
    circle(circles([1 2],q),circles(3,(q)),16,[0 1 0]);
end

end

