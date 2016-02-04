function [feats,labels] = samplesToFeats(samples,featureExtractor)
%UNTITLED5 Summary of this function goes here
%   Detailed explanation goes here
feats = {};
labels = {};
patches = {};
tic_id = ticStatus('extracting features from samples...',.5,.5);
for t = 1:length(samples)
    masks = samples(t).masks;
    n = length(masks);
    I = samples(t).img;
    labels{t} = ones(1,n)*samples(t).label;    
    %feats{t} = featureExtractor.extractFeaturesMulti_mask(I,masks);
    patches{end+1} = cellfun2(@(x) maskedPatch(I,x,true,.5),masks);
    tocStatus(tic_id,t/length(samples));
end

labels = cat(2,labels{:});
patches = cat(2,patches{:});
feats = featureExtractor.extractFeaturesMulti(patches);
geom_descs = cat(2,samples.geom_descs);
feats = [feats;geom_descs];
end

