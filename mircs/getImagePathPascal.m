function [ imagePath ] = getImagePathPascal(conf,currentID)
%GETIMAGEFILE Summary of this function goes here
%   Detailed explanation goes here
if (any(strfind(currentID,'.jpg')))
    currentID = strrep(currentID,'.jpg','');
end
imagePath = sprintf(conf.VOCopts.imgpath,currentID);
end