function [bbox] = inflatebbox(bbox,inflation)
%INFLATEBBOX Summary of this function goes here
%   Detailed explanation goes here
xmin = bbox(1);
ymin= bbox(2);
xmax = bbox(3);
ymax = bbox(4);

centerx = (xmax+xmin)/2;
centery = (ymax+ymin)/2;

w = (xmax-xmin)*inflation;
h = (ymax-ymin)*inflation;
xmin = centerx-w/2;
ymin = centery-h/2;
xmax = centerx+w/2;
ymax = centery+h/2;

bbox = [xmin ymin xmax ymax];


end

