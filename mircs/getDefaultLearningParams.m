
function [learnParams,conf] = getDefaultLearningParams(conf,nWords)
if (nargin < 2)
    nWords = 256;
end
learnParams.doHardNegativeMining = false;
learnParams.nNegativesPerPositive = 20;
learnParams.nNegativeMiningRounds = 3;
learnParams.balanceDatasets = true;
learnParams.useRealGTSegments = false;
learnParams.hkmGamma = 1;
conf.featConf = init_features(conf,nWords);
% learn the part models...
extractors = initializeFeatureExtractors(conf);
learnParams.featureExtractors = extractors;
learnParams.debugSuffix = '3_256_pn';