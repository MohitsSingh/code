
function params = defaultPipelineParams(calcFeats)
if nargin < 1
    calcFeats = true;
end
% region of interest
roiParams = defaultROIParams();
% saliency options
saliencyOpts.maxImageSize = 200;
saliencyOpts.pixNumInSP = 50;
saliencyOpts.show = false;
params.roiParams = roiParams;
params.saliencyParams = saliencyOpts;
% learning parameters
params.learning.nNegsPerPos = 200;
params.learning.negOvp = .3;
params.learning.posOvp = .55;
params.learning.maxNegsToKeep = inf;
params.debug = false;
params.skipCalculation = false;
params.segmentation.useGraphCut = false;%{'ucm','graph-cut'};
params.testMode = true;
params.keepSegments = true;
params.get_gt_regions = true;
% which features to extract?
% what keypoints to extract in each face?
params.requiredKeypoints = {'LeftEyeCenter','RightEyeCenter','MouthCenter','MouthLeftCorner','MouthRightCorner','ChinCenter','NoseCenter'};
params.features.getLineSegs = true;
params = resetFeatures(params,calcFeats);
params.dataDir = [];
params.prevStageDir = [];
