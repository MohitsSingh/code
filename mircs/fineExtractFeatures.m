function feats = fineExtractFeatures( imgData,regions, phase )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
feats = phase.featureExtractor.extractFeaturesMulti_mask(imgData.I_sub,regions);
end

