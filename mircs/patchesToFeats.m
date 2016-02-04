function [pos_feats,neg_feats] = patchesToFeats(pos_patches,neg_patches,sz);
if (nargin < 3)
    sz = [64 64];
end
if (isscalar(sz))
    sz = [sz sz];
end
pos_feats = getFeats(pos_patches,sz);
if (isempty(neg_patches))
    neg_feats = [];
else
    neg_feats = getFeats(neg_patches,sz);
end

    function feats = getFeats(patches,sz)
        
        
        
        patches = cellfun2(@(x) imResample(x,sz,'bilinear'),patches);
%         feats = imageSetFeatures2(conf,patches,true);
        feats = piotr_features2(patches);
        feats = cellfun2(@col,feats);
        feats = cat(2,feats{:});
    end
end