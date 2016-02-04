function [xy] = inflatePolygon2(xy,f)
[x,y] = inflatePolygon(xy(:,1),xy(:,2),f);
xy = [x,y];

end