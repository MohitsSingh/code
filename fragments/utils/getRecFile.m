function [recPath] = getRecFile(globalOpts,imageID)
%GETRECFILE Summary of this function goes here
%   Detailed explanation goes here
    recPath = sprintf(globalOpts.VOCopts.annopath,imageID);

end

