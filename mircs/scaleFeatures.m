function [normalizer,feats] = scaleFeatures(feats,normalizer)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if nargin < 2
        min_val = min(feats,[],2);
        max_val = max(feats,[],2);
        assert(all(max_val>min_val));
        normalizer.min_val = min_val;
        normalizer.max_val = max_val;
    end
    
    if nargout > 1
        min_val = normalizer.min_val;
        max_val = normalizer.max_val;
        feats = bsxfun(@minus,feats,min_val);
        feats = bsxfun(@rdivide,feats,max_val-min_val);
    end
end

