function [f_int,f_app,f_shape] = featureHelper(I, regions, featureExtractor)
regions = row(regions);
f_int = zeros(size(regions));
f_app = featureExtractor.extractFeaturesMulti_mask(I,regions,true);
s_masks = cellfun2(@(x) repmat(x,[1 1 3]),regions);
f_shape = featureExtractor.extractFeaturesMulti(s_masks,true);