function params = resetFeatures(params,calcFeats)
% reset feature parameters to compute the default features, unless
% calcFeats is false, in which case nothing will be calculated.

% geometric/interaction features
params.features.getBoxFeats = true;
params.features.getSimpleShape = true & calcFeats;
params.features.getGeometry = true & calcFeats;
params.features.getGeometryLogPolar = true  & calcFeats;
params.features.getShape = true  & calcFeats;
params.features.getLogPolarShape = false  & calcFeats;
params.features.getPixelwiseProbs = true  & calcFeats;

% appearance features
params.features.getHOGShape = true & calcFeats;
params.features.getAppearance = false & calcFeats;
params.features.getAppearanceDNN = true & calcFeats;