function [I] = readImage(VOCopts,imageID)
%GETIMAGEPATH Summary of this function goes here
%   Detailed explanation goes here
    I = imread(sprintf(VOCopts.imgpath,imageID));

end

