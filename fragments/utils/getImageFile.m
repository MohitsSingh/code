function [ imagePath ] = getImageFile(globalOpts,currentID)
%GETIMAGEFILE Summary of this function goes here
%   Detailed explanation goes here
imagePath = sprintf(globalOpts.VOCopts.imgpath,currentID);
end

