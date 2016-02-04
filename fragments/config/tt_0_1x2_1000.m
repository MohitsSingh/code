defaultOpts;

% custom opts
globalOpts.numTrain = inf;
globalOpts.det_rescale = 0;
globalOpts.use_overlapping_negatives = true;
globalOpts.scale_choice = 4;
globalOpts.numSpatialX = [1 2];
globalOpts.numSpatialY = [1 2]
globalOpts.debug  = 0;
globalOpts.removeOverlappingDetections = true;
globalOpts.maxTrain = inf;
globalOpts.numTrain = inf;
% globalOpts.phowOpts = {'Step',1};
globalOpts.numWords = 1000;



% update globalOpts to reflect custom options.
updateOpts;
