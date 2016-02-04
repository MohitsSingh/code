function [patches,context_feats_mean,context_feats_max] = extractContextFeatures(I,probs,boxes)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
patches = multiCrop2(I,boxes);
boxes = round(inflatebbox(boxes,3,'both',false));
context_feats = multiCrop2(probs,boxes);

context_feats_max = cellfun2(@(x) max(max(x,[],1),[],2),context_feats);

% context_feats_max = max(max(cat(3,context_feats{:}),[],1),[],2);
context_feats_mean = multiResize(context_feats,[5 5]);

end

