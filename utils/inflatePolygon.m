function [x,y] = inflatePolygon(x,y,f)
mean_x = mean(x);
mean_y = mean(y);
x = (x-mean_x)*f+mean_x;
y = (y-mean_y)*f+mean_y;
end