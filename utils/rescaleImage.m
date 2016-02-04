function [im,scaleFactor] = rescaleImage(im,maxHeight,enlarge)

if (nargin < 3)
    enlarge = false;
end
if (enlarge || size(im,1) > maxHeight)
    scaleFactor = maxHeight/size(im,1);
else
    scaleFactor = 1;
end
im = imResample(im,scaleFactor,'bilinear');