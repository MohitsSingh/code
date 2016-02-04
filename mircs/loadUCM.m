function [ucm,gPb_thin] = loadUCM(conf,currentID,gpbDir) %#ok<*STOUT>

if (nargin == 3)
    resDir = gpbDir;
else
    resDir = conf.gpbDir;
end
ucmFile = fullfile(resDir,strrep(currentID,'.jpg','_ucm.mat'));
load(ucmFile); % ucm
if (nargout == 2)
    gpbFile = fullfile(resDir,strrep(currentID,'.jpg','.mat'));
    load(gpbFile);
end
end