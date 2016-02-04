function [bbox] = inflatebbox(bbox,inflation,direction,absFlag)
%INFLATEBBOX Summary of this function goes here
%   Detailed explanation goes here
if (size(inflation,2)==1)
    inflation = [inflation inflation];
end
if (size(inflation,1)==1)
    inflation = repmat(inflation,size(bbox,1),1);
end

if (size(bbox,2)==2)
    bbox = [bbox bbox];
end

if (nargin < 4)
    absFlag = false;
end

xmin = bbox(:,1);
ymin = bbox(:,2);
xmax = bbox(:,3);
ymax = bbox(:,4);

centerx = (xmax+xmin)/2;
centery = (ymax+ymin)/2;
if (nargin < 3)
    direction = 'both';
end

if (isscalar(inflation))
    inflation = [inflation inflation];
end

w = (xmax-xmin+1).*inflation(:,1);
h = (ymax-ymin+1).*inflation(:,2);
if (absFlag)
    w = inflation(:,1);
    h = inflation(:,2);
end
if (strcmp(direction,'pre'))
    xmin = centerx-w/2;
    ymin = centery-h/2;
elseif (strcmp(direction,'post'))
    xmax = xmin+w-1;
    ymax = ymin+h-1;
else
    xmin = centerx-w/2;
    ymin = centery-h/2;
    xmax = xmin+w;
    ymax = ymin+h;
end

bbox = [xmin ymin xmax ymax];

end

