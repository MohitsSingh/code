function [ featPath ] = getFeatFile(globalOpts,imageID)
%GETFEATFILE Summary of this function goes here
%   Detailed explanation goes here
    featPath = sprintf(globalOpts.featPath,imageID);

end

