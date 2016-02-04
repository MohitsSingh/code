function feats = coarseExtractFeatures( imgData,regions, phase )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    feats = phase.featureExtractor.extractFeaturesMulti_mask(imgData.I_sub,regions);
end

