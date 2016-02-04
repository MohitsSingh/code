function [ scores,feats ] = scoreRegions(I,samples,featureExtractor,w,b,varargin)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
ip = inputParser;
ip.addParameter('mask_patches',true,@islogical);
ip.addParameter('crop_patches',true,@islogical);
ip.addOptional('normalizer',[]);

ip.parse(varargin{:});
masks = samples.masks;
mask_patches = ip.Results.mask_patches;
normalizer = ip.Results.normalizer;
geom_feats = cat(2,samples.geom_descs);
% [feats,labels] = samplesToFeats(samples,featureExtractor)

if mask_patches
    patches = cellfun2(@(x) maskedPatch(I,x,true,.5),masks);
else % just crop around them
    rects = cellfun3(@(x) round(makeSquare(region2Box(x),true)), masks);
    patches = multiCrop2(I,rects);
    %patches = cellfun2(@(x) cropper(I,);    
end
feats = featureExtractor.extractFeaturesMulti(patches,true);
feats = [feats;geom_feats];
if ~isempty(normalizer)
    [~,feats] = scaleFeatures(feats,normalizer);
end
scores = w'*feats+b;

end

