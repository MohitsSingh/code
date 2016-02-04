function [imagePath] = getImagePath(conf,imageID)
%GETIMAGE Summary of this function goes here
%   Detailed explanation goes here
if (exist(imageID,'file'))
    imagePath = imageID;
else
    imagePath = fullfile(conf.imgDir,imageID);
    if (~exist(imagePath,'file'))
        imagePath = getImagePathPascal(conf,imageID);
    end
end

