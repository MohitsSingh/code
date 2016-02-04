function [bbox] = inflatebbox(bbox,inflation)
%INFLATEBBOX Summary of this function goes here
%   Detailed explanation goes here
xmin = bbox(1);
ymin= bbox(2);
xmax = bbox(3);
ymax = bbox(4);

centerx = (xmax+xmin)/2;
centery = (ymax+ymin)/2;

w = xmax-xmin;
h = ymax-ymin;
xmin = centerx-(1+inflation)*w/2;
ymin = centery-(1+inflation)*h/2;
xmax = centerx+(1+inflation)*w/2;
ymax = centery+(1+inflation)*h/2;

bbox = [xmin ymin xmax ymax];


end

