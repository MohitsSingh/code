function [boxesFile] = getBoxesFile(globalOpts,imageID)
%GETBOXESFILE Summary of this function goes here
%   Detailed explanation goes here
boxesFile  =sprintf(globalOpts.VOCopts.exfdpath,[imageID '_boxes']);

end

