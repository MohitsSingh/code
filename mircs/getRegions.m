function [candidates,ucm2,success] = getRegions(conf,currentID,segDir)
if (nargin < 3)
    segDir = '~/storage/s40_seg_new';
end
segPath = j2m(segDir,currentID);
load(segPath);