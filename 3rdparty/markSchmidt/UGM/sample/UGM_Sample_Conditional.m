function [samples] = UGM_Sample_Conditional(nodePot,edgePot,edgeStruct,clamped,sampleFunc,varargin)
% Do sampling with observed values

[nNodes,maxState] = size(nodePot);
nEdges = size(edgePot,3);
edgeEnds = edgeStruct.edgeEnds;
maxIter = edgeStruct.maxIter;

[clampedNP,clampedEP,clampedES,edgeMap] = UGM_makeClampedPotentials(nodePot,edgePot,edgeStruct,clamped);

clampedSamples = sampleFunc(clampedNP,clampedEP,clampedES,varargin{:});

% Construct node beliefs
samples = repmat(clamped,[1 maxIter]);
samples(samples==0) = clampedSamples;